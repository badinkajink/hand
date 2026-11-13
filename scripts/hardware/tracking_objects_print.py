#!/usr/bin/env python3
"""Print files of the tracked objects of the object seating and tracking plan (docs/notes/20261105-object_seating_and_
tracking/): the 25 mm x 100 mm cylinder and the 40 mm sphere of the HOM tasks, each with its hexagonal tag block on a
stem below it (owner, 2026-11-12: stems for both layout sets, grip tape of about 0.8 mm on the cylinder, the sphere bare,
the sphere at the plan's two clear layouts).

Each object is printed in parts that are glued: the block and its stem in one piece, upright like the test block of
2026-11-07, so the stem lies on the block's axis; the stem ends in a 45 deg cone, a flat seat SEAT_D across and a
rectangular key, which sit in a matching flat and slot on the object and fix its turn about the stem. The cylinder is
printed standing, so it comes out round, at the taped diameter less twice the tape; its end faces bear tag36h11 ids 22
and 23 at 13 mm, so a bundle adjustment of the assembled object measures the cylinder against its block and the glue
joint's error drops out. The sphere is printed as two hemispheres, flat side down, joined by three dowels. The geometry
is object_tracking_mjcf.OBJECTS and object_map, the tracker's: the block's top lies stem_mm below the object's taped or
bare surface (rows/stem_sv25_sv55db.json: 15 mm below the lowest finger point of the plan's five layouts and the measured
servo's five best per task).

Files lie in print orientation on z = 0, each two-colour part as a white body and its black cells in one frame. `build`
(femtip, manifold3d) writes them; `check` (camera stack) draws every tag face of the black bodies and detects it against
the map; `bambu` writes one Bambu Studio project of all parts (white on filament 2, black on 3). `build` also writes two
alternatives of the cylinder, with end tag 23 only and without end tags (owner, 2026-11-13: the end tags "cause a lot of
pla usage"), and `bambu --cylinder NAME` slices the plate with one of them in a project of its own.

    ~/.claude/bin/resguard.sh run --mem 3G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/envs/femtip/bin/python scripts/hardware/tracking_objects_print.py build
    ~/.claude/bin/resguard.sh run --mem 2G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/bin/python scripts/hardware/tracking_objects_print.py check [--preview PNG]
    ~/.claude/bin/resguard.sh run --mem 3G -- env NUMPY_MADVISE_HUGEPAGE=0 \\
        ~/miniconda3/bin/python scripts/hardware/tracking_objects_print.py bambu
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
import object_tracking_print as OP  # noqa: E402
from sr2hand.tracking import tags as TG  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATE = "20261112"
OUT = ROOT / f"assets/tracking/{DATE}-objects"
REPORT = OUT / f"{DATE}-objects.json"
PROJECT = OUT / f"{DATE}-objects_p1s.3mf"
SEAT_D, CONE_H = 10.0, 3.0      # mm: the flat seat between stem and object, and the 45 deg cone up to it
KEY = (2.4, 5.0, 4.0)           # mm: the key on the seat along x_B, y_B and up
KEY_CLEAR, SLOT_EXTRA = 0.1, 0.5   # mm: the slot's clearance per side and its depth beyond the key
END_DEPTH = 0.6                 # mm: the black of the cylinder's end tags, three 0.2 mm layers
PIN_D, PIN_HOLE_D, PIN_R, PIN_DEPTH = 2.9, 3.1, 10.0, 4.0   # mm: the sphere's dowels, their holes, radius, depth per half
PIN_ANGLES = (90.0, 210.0, 330.0)
SEG = 160                       # segments of the round surfaces
PLA, MASS_G = 1.24e-3, 25.0     # g/mm^3; the HOM objects' mass with stem, block and tape
# O to print: the cylinder stands on its -z_O end; the hemispheres lie on the equator, the lower one pole up
R_PRINT = {"cylinder": np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]),
           "sphere_lower": np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]),
           "sphere_upper": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])}


def seat(shape):
    """(printed radius, depth of the flat seat below the printed surface, the seat plane's y_O) of an object."""
    o = OM.OBJECTS[shape]
    r = o["r_mm"] - o["tape_mm"]
    depth = r - math.sqrt(r * r - (SEAT_D / 2) ** 2)
    return r, depth, -(r - depth)


def m4(R, t=(0.0, 0.0, 0.0)):
    return np.hstack([np.asarray(R, float), np.asarray(t, float)[:, None]])


def block_stem(shape):
    """(white, black) of the block and its stem in the block frame B moved up by half the block (print frame)."""
    import manifold3d as M
    B, o = OM.BLOCK, OM.OBJECTS[shape]
    _r, _d, seat_y = seat(shape)
    z_seat = B.height / 2 + seat_y + (o["r_mm"] + o["stem_mm"])      # the seat plane in B: the block's top is r + stem below O
    prism = M.Manifold.extrude(M.CrossSection([B.vertices()[:6, :2].tolist()]), B.height).translate(
        [0.0, 0.0, -B.height / 2])
    stem = M.Manifold.cylinder(z_seat - CONE_H - B.height / 2 + 0.5, OM.STEM_D / 2, OM.STEM_D / 2, SEG).translate(
        [0.0, 0.0, B.height / 2 - 0.5])
    cone = M.Manifold.cylinder(CONE_H, OM.STEM_D / 2, SEAT_D / 2, SEG).translate([0.0, 0.0, z_seat - CONE_H])
    key = M.Manifold.cube([KEY[0], KEY[1], KEY[2]]).translate([-KEY[0] / 2, -KEY[1] / 2, z_seat])
    S = M.Manifold.batch_boolean([prism, stem, cone, key], M.OpType.Add)
    cells = OP.black_cells(OM.block_map(shape), OP.BLOCK_DEPTH)
    lift = np.eye(3), (0.0, 0.0, B.height / 2)
    return (S - cells).transform(m4(*lift)), (S ^ cells).transform(m4(*lift)), z_seat + B.height / 2


def slot(seat_y):
    """The key's slot in the object, from the seat plane into +y_O, its long side along z_O (= -y_B)."""
    import manifold3d as M
    w, l, d = KEY[0] + 2 * KEY_CLEAR, KEY[1] + 2 * KEY_CLEAR, KEY[2] + SLOT_EXTRA
    return M.Manifold.cube([w, d + 0.5, l]).translate([-w / 2, seat_y - 0.5, -l / 2])


def flat(r, seat_y):
    """What the flat seat removes below the plane y_O = seat_y: a cylinder SEAT_D across along y_O."""
    import manifold3d as M
    h = seat_y + r + 1.0
    return M.Manifold.cylinder(h, SEAT_D / 2 + 0.01, SEAT_D / 2 + 0.01, SEG).transform(
        m4(np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]), (0.0, -r - 1.0, 0.0)))  # its axis along +y_O, up to seat_y


def cylinder_body(ids=OM.END_IDS):
    """(white, black) of the cylinder in O turned to print, standing on its -z_O end, with the end tags `ids`; black is
    None when `ids` is empty."""
    import manifold3d as M
    r, _d, seat_y = seat("cylinder")
    L = OM.OBJECTS["cylinder"]["length_mm"]
    S = M.Manifold.cylinder(L, r, r, SEG).translate([0.0, 0.0, -L / 2]) - flat(r, seat_y) - slot(seat_y)
    T = m4(R_PRINT["cylinder"], (0.0, 0.0, L / 2))
    if not ids:
        return S.transform(T), None
    ends = TG.TagMap(OM.BLOCK.family, OM.BLOCK.tag_size)
    full = OM.object_map("cylinder")
    for i in ids:
        ends.frames[i], ends.corners[i] = full.frames[i], full.corners[i]
    cells = OP.black_cells(ends, END_DEPTH)
    return (S - cells).transform(T), (S ^ cells).transform(T)


def sphere_halves():
    """(lower, upper) hemispheres of the sphere, each on its equator face (print frames), and one dowel pin."""
    import manifold3d as M
    r, _d, seat_y = seat("sphere")
    ball = M.Manifold.sphere(r, SEG)
    big = 2 * r + 4
    low = ball ^ M.Manifold.cube([big, big / 2, big]).translate([-big / 2, -big / 2, -big / 2])
    up = ball ^ M.Manifold.cube([big, big / 2, big]).translate([-big / 2, 0.0, -big / 2])
    holes = []
    for a in PIN_ANGLES:
        x, z = PIN_R * math.cos(math.radians(a)), PIN_R * math.sin(math.radians(a))
        holes.append(M.Manifold.cylinder(2 * PIN_DEPTH, PIN_HOLE_D / 2, PIN_HOLE_D / 2, 48).transform(
            m4(np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]), (x, -PIN_DEPTH, z))))  # along y_O, PIN_DEPTH each side
    H = M.Manifold.batch_boolean(holes, M.OpType.Add)
    low = low - H - flat(r, seat_y) - slot(seat_y)
    up = up - H
    pin = M.Manifold.cylinder(2 * PIN_DEPTH - 0.4, PIN_D / 2, PIN_D / 2, 48)
    return low.transform(m4(R_PRINT["sphere_lower"])), up.transform(m4(R_PRINT["sphere_upper"])), pin


def stl(name, colour="white"):
    return OUT / f"{DATE}-{name}_{colour}.stl"


PARTS = {"cylinder_block": "cylinder block and stem, ids 10-15", "sphere_block": "sphere block and stem, ids 16-21",
         "cylinder": "cylinder body, end tags 22 and 23", "sphere_lower": "sphere, lower hemisphere with the seat",
         "sphere_upper": "sphere, upper hemisphere", "sphere_pin": "dowel pin of the sphere (print three)"}
# the cylinder with fewer end tags (owner, 2026-11-13: "do we need the apriltag on the end of the cylinder? it causes a
# lot of pla usage"). Tag 22 lies on the printed cylinder's top, so with it the prime tower rises to 100 mm; tag 23
# prints in the first three layers. Each alternative replaces the cylinder on the plate (bambu --cylinder).
CYLINDERS = {"cylinder": OM.END_IDS, "cylinder_tag23": (23,), "cylinder_plain": ()}
ALTERNATIVES = {"cylinder_tag23": "cylinder body, end tag 23 only", "cylinder_plain": "cylinder body without end tags"}
PRINTED = {"cylinder": "cylinder_plain"}   # owner, 2026-11-13: "i sent the print with no cylinder end tag"


def cmd_build(a):
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"what": "the tracked cylinder and sphere with their tag blocks, as print files",
           "script": "scripts/hardware/tracking_objects_print.py", "objects": OM.OBJECTS,
           "seat": {s: dict(zip(("r_print_mm", "flat_depth_mm", "seat_y_O_mm"), map(float, seat(s))))
                    for s in ("cylinder", "sphere")},
           "parts": {}}
    if REPORT.exists():                   # the projects' slices stay until bambu slices again
        old = json.loads(REPORT.read_text())
        rep.update({k: old[k] for k in ("bambu", "bambu_variants") if k in old})
    bodies = {}
    for shape in ("cylinder", "sphere"):
        w, b, top = block_stem(shape)
        bodies[f"{shape}_block"] = (w, b)
        rep["parts"][f"{shape}_block"] = {"height_mm": round(top + KEY[2], 3)}
    bodies.update({name: cylinder_body(ids) for name, ids in CYLINDERS.items()})
    low, up, pin = sphere_halves()
    # the assembly: each block and stem moved back into O against its body; the seat only touches the flat and the key
    # clears its slot, so the shared volume is zero
    rep["assembly_overlap_mm3"] = {}
    for shape, body, R in (("cylinder", bodies["cylinder"][0] + bodies["cylinder"][1], R_PRINT["cylinder"]),
                           ("sphere", low, R_PRINT["sphere_lower"])):
        o = OM.OBJECTS[shape]
        w, b = bodies[f"{shape}_block"]
        blk = (w + b).transform(m4(OM.R_OB, OM.block_centre_O(o["r_mm"], o["stem_mm"])
                                   - OM.R_OB @ np.array([0.0, 0.0, OM.BLOCK.height / 2])))
        shift = np.array([0.0, 0.0, -o["length_mm"] / 2 if shape == "cylinder" else 0.0])
        rep["assembly_overlap_mm3"][shape] = abs(round((blk ^ body.transform(m4(R.T, shift))).volume(), 6))
        hexagon = 3 * OM.BLOCK.face_width * OM.BLOCK.across_flats / 2 * OM.BLOCK.height
        rep["parts"][f"{shape}_block"]["stem_solid_g"] = round(((w + b).volume() - hexagon) * PLA, 2)
    print("assembly overlap (mm3)", rep["assembly_overlap_mm3"])
    bodies.update(sphere_lower=(low, None), sphere_upper=(up, None), sphere_pin=(pin, None))
    ok = True
    for name, (w, b) in bodies.items():
        info = rep["parts"].setdefault(name, {})
        info["what"] = {**PARTS, **ALTERNATIVES}[name]
        vol = {"white": w.volume(), "black": b.volume() if b is not None else 0.0}
        if b is not None:
            vol["overlap"] = (w ^ b).volume()
            ok &= vol["overlap"] < 1e-6
        info["volume_mm3"] = {k: round(v, 3) for k, v in vol.items()}
        info["solid_pla_g"] = round((vol["white"] + vol["black"]) * PLA, 2)
        for colour, m in (("white", w), ("black", b)):
            if m is None:
                continue
            t = OP.to_trimesh(m)
            good = bool(t.is_watertight and t.is_winding_consistent and t.volume > 0)
            ok &= good
            t.export(stl(name, colour))
            info[colour] = {"file": str(stl(name, colour).relative_to(ROOT)), "triangles": int(len(t.faces)),
                            "watertight": good, "shells": len(t.split(only_watertight=False)),
                            "bounds_mm": np.round(t.bounds, 3).tolist()}
        print(name, "ok" if ok else "FAILED", info["volume_mm3"], info.get("white", {}).get("bounds_mm"), flush=True)
    rep["ok"] = bool(ok)
    REPORT.write_text(json.dumps(rep, indent=1))
    print("wrote", REPORT.relative_to(ROOT))
    if not ok:
        raise SystemExit(1)


def print_maps():
    """{part: [TagMap in its print frame]} of the two-colour parts."""
    out = {}
    for shape in ("cylinder", "sphere"):
        out[f"{shape}_block"] = [OP.shifted(OM.block_map(shape), OM.BLOCK.height / 2)]
    full = OM.object_map("cylinder").transformed(np.r_[m4(R_PRINT["cylinder"], (0.0, 0.0, 50.0)), [[0, 0, 0, 1.0]]])
    for name, ids in CYLINDERS.items():
        if ids:
            ends = TG.TagMap(full.family, full.size)
            for i in ids:
                ends.frames[i], ends.corners[i] = full.frames[i], full.corners[i]
            out[name] = [ends]
    return out


def cmd_check(a):
    import trimesh
    from sr2hand.tracking import pose as TP
    rep = json.loads(REPORT.read_text())
    tiles, ok = [], True
    for part, tms in print_maps().items():
        mesh = trimesh.load(stl(part, "black"), force="mesh")
        rows = []
        for tm in tms:
            det = TP.detector(tm.family)
            for tid, (c, r, d, n) in tm.frames.items():
                cell = OP.edge(tm, tid) / TG.n_cells(tm.family)
                half = OP.edge(tm, tid) / 2 + 2 * cell
                img = OP.draw(OP.in_plane(mesh, c, n), c, r, d, half)
                found = TP.detect(det, img)
                want = OP.to_px(tm.corners[tid], c, r, d, half)
                err = float(np.sqrt(((found[tid] - want) ** 2).sum(1).mean())) / OP.PPM if tid in found else None
                rows.append({"id": tid, "found": sorted(found), "corner_rms_mm": None if err is None else round(err, 4)})
                tiles.append((f"{part} {tid}", img, found.get(tid), want))
                ok &= sorted(found) == [tid] and err is not None and err < 0.05
        rep["parts"][part]["check"] = rows
        print(part, rows, flush=True)
    rep["check_ok"] = bool(ok)
    REPORT.write_text(json.dumps(rep, indent=1))
    if a.preview:
        OP.sheet(tiles, a.preview)
    write_readme(rep)
    print("check", "ok" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


# on the P1S bed (mm): each part's white body centred here; the prime tower clear of all
PLACE = {"cylinder_block": (45.0, 60.0), "sphere_block": (45.0, 110.0), "cylinder": (45.0, 165.0),
         "sphere_lower": (110.0, 60.0), "sphere_upper": (110.0, 120.0), "sphere_pin": (150.0, 60.0)}
TOWER = (190.0, 180.0)


def cmd_bambu(a):
    items = []
    for name in PARTS:
        src = a.cylinder if name == "cylinder" else name
        it = {"name": name, "white": stl(src), "black": stl(src, "black") if stl(src, "black").exists() else None,
              "place": PLACE[name]}
        items.append(it)
    pins = [dict(items[-1], name=f"sphere_pin_{k}", place=(PLACE["sphere_pin"][0] + 12.0 * k, PLACE["sphere_pin"][1]))
            for k in range(3)]
    items = items[:-1] + pins
    variant = a.cylinder != "cylinder"
    project = OUT / f"{DATE}-objects_p1s_{a.cylinder}.3mf" if variant else PROJECT
    work = ROOT / f"logs/{DATE}-objects/bambu{'_' + a.cylinder if variant else ''}"
    q = OP.bambu_project(items, project, work, "tracked objects", tower=TOWER)
    rep = json.loads(REPORT.read_text())
    if variant:
        rep.setdefault("bambu_variants", {})[a.cylinder] = q
    else:
        rep["bambu"] = q
    REPORT.write_text(json.dumps(rep, indent=1))
    write_readme(rep)
    print(f"{project.relative_to(ROOT)}: {q['prediction_s'] / 3600:.1f} h, {q['layers']} layers, "
          f"{q['filament_loads']} filament loads, used {q['used_g']} g")
    print("problems:", q["problems"] or "none")
    if q["problems"]:
        raise SystemExit(1)


def write_readme(rep):
    P, sc, o = rep["parts"], rep["seat"], OM.OBJECTS
    blk = {s: OM.BLOCK_G + P[f"{s}_block"]["stem_solid_g"] for s in ("cylinder", "sphere")}
    L = [f"# Tracked cylinder and sphere, {DATE}",
         "",
         "Print files of the objects of `docs/notes/20261105-object_seating_and_tracking/`, written by "
         "`scripts/hardware/tracking_objects_print.py`; the tracker's maps of the assembled objects are "
         "`object_tracking_mjcf.object_map(shape)`. Two-colour parts are a white body and its black cells in one "
         "frame: load both files at once as one object with two parts and give the black part the black filament.",
         "",
         "| Part | Files | Print |",
         "|---|---|---|"]
    names = list(PARTS)
    k = names.index("cylinder") + 1
    names[k:k] = [n for n in ALTERNATIVES if n in P]
    for name in names:
        what, p = {**PARTS, **ALTERNATIVES}[name], P[name]
        files = ", ".join(f"`{Path(p[c]['file']).name}`" for c in ("white", "black") if c in p)
        how = {"cylinder_block": "upright, block on the bed; 65 layers in both colours",
               "sphere_block": "upright, block on the bed; 65 layers in both colours",
               "cylinder": "standing on its -z end (tag 23 on the bed, tag 22 on top)",
               "cylinder_tag23": "in place of the cylinder above, standing on its -z end (tag 23 on the bed)",
               "cylinder_plain": "in place of the cylinder above, standing; white only",
               "sphere_lower": "flat side down, the seat on top", "sphere_upper": "flat side down",
               "sphere_pin": "standing; print three"}[name]
        L.append(f"| {what} | {files} | {how} |")
    L += ["",
          f"Geometry. The cylinder prints at {2 * sc['cylinder']['r_print_mm']:.1f} mm, {o['cylinder']['length_mm']:g} mm "
          f"long, so {o['cylinder']['tape_mm']} mm grip tape brings it to {2 * o['cylinder']['r_mm']:g} mm; the end faces "
          f"stay bare for their tags. The sphere is {2 * o['sphere']['r_mm']:g} mm and bare. The block's top lies "
          f"{o['cylinder']['stem_mm']} mm (cylinder) and {o['sphere']['stem_mm']} mm (sphere) below the object's taped or "
          "bare surface, 15 mm below the lowest finger point of the plan's five layouts and the measured servo's five "
          f"best per task. The flat seat is {SEAT_D:g} mm across and lies {sc['cylinder']['flat_depth_mm']:.2f} mm "
          f"(cylinder) and {sc['sphere']['flat_depth_mm']:.2f} mm (sphere) below the printed surface; the key is "
          f"{KEY[0]} x {KEY[1]} x {KEY[2]} mm with {KEY_CLEAR} mm clearance per side in its slot, its long side along "
          "the cylinder's axis.",
          "",
          "Assembly. Glue the key into its slot with the seat flat on the object's flat (thin CA glue or epoxy), the "
          "block's tag faces then upright. Glue the hemispheres with the three dowels. Wrap the cylinder's curved "
          f"surface in the tape, leaving a {SEAT_D:g} mm hole at the seat. Weigh: each object with its block and tape is "
          f"{MASS_G:g} g in the simulations. The plan expects {OM.BLOCK_G:g} g for a block with 0.8 mm walls, and the stem "
          f"with its cone and key is {P['cylinder_block']['stem_solid_g']} g (cylinder) and "
          f"{P['sphere_block']['stem_solid_g']} g (sphere) of solid PLA, so the cylinder body with its tape has about "
          f"{MASS_G - blk['cylinder']:.1f} g and the sphere about {MASS_G - blk['sphere']:.1f} g. Weigh the printed blocks "
          "and the tape first, then set the bodies' infill to make up the rest.",
          "",
          "Calibration. A bundle adjustment of the assembled cylinder over its eight tags (`pose.bundle_adjust` with "
          "the block's three across-flats as lengths) gives its map, and with it where the cylinder sits on its block. "
          "The sphere's map is its block's, placed by the seat and key. In the CAD the assembled parts share "
          f"{rep['assembly_overlap_mm3']['cylinder']:g} mm3 (cylinder) and {rep['assembly_overlap_mm3']['sphere']:g} mm3 "
          "(sphere): the seats touch the flats and the keys clear their slots. With end tag 23 alone the adjustment has "
          "seven tags, and the cylinder's turn about the stem is read from where tag 23 sits, "
          f"{o['cylinder']['length_mm'] / 2:g} mm from the stem. Without end tags the cylinder's map is the CAD's; the "
          f"key's clearance then leaves its turn about the stem within +-{math.degrees(2 * KEY_CLEAR / KEY[1]):.1f} deg.",
          "",
          f"Printed: the owner printed the {ALTERNATIVES[PRINTED['cylinder']].replace('body, ', '')} on 2026-11-13. "
          "Its map is then its block's, placed by the seat and key, as for the sphere. A flat plate pressed on the face "
          f"of tag {OM.IDS['cylinder'][0]} or {OM.IDS['cylinder'][3]} lies parallel to the cylinder's axis, "
          f"{OM.BLOCK.across_flats / 2 - sc['cylinder']['r_print_mm']:.1f} mm outside its printed side. Equal gaps "
          "at both ends of the cylinder set its turn on the block before the glue sets (epoxy leaves time to "
          "adjust). A gap difference d over a spacing L along the axis is a turn of d / L rad."]
    if "check_ok" in rep:
        rms = [r["corner_rms_mm"] for p in P.values() for r in p.get("check", [])]
        span = f"{min(rms):.3f}" if f"{min(rms):.3f}" == f"{max(rms):.3f}" else f"{min(rms):.3f} to {max(rms):.3f}"
        L += ["", f"Check: every tag face of the black bodies, drawn at {OP.PPM:.0f} px/mm from the mesh, decodes as its "
                  f"own id with corners {span} mm RMS from the map."]
    def sliced(q):
        return (f"{q['prediction_s'] / 3600:.1f} h, {q['used_g'].get('2', 0):.1f} g white and "
                f"{q['used_g'].get('3', 0):.1f} g black, {q['filament_loads']} filament loads")

    if "bambu" in rep:
        q = rep["bambu"]
        s = (f"Bambu Studio project: `{Path(q['project']).name}`, all parts on one plate, the white on filament 2 and "
             f"the black on filament 3 (AMS slots 2 and 3). Sliced by the command line: {sliced(q)}.")
        phrase = {"cylinder_tag23": "the cylinder bearing only end tag 23", "cylinder_plain": "the cylinder without end tags"}
        if rep.get("bambu_variants"):
            s += (" Tag 22 lies on the printed cylinder's top, so with it the prime tower rises to "
                  f"{o['cylinder']['length_mm']:g} mm.")
        for v, qv in rep.get("bambu_variants", {}).items():
            s += f" With {phrase[v]} in its place, `{Path(qv['project']).name}`: {sliced(qv)}."
        L += ["", s + " Not in git; `tracking_objects_print.py bambu [--cylinder NAME]` rebuilds them."]
    (OUT / "README.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p = sub.add_parser("check")
    p.add_argument("--preview")
    p = sub.add_parser("bambu")
    p.add_argument("--cylinder", choices=list(CYLINDERS), default="cylinder",
                   help="the cylinder on the plate; an alternative writes its own project beside the default one")
    a = ap.parse_args()
    {"build": cmd_build, "check": cmd_check, "bambu": cmd_bambu}[a.cmd](a)


if __name__ == "__main__":
    main()
