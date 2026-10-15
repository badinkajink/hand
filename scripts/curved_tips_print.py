#!/usr/bin/env python3
"""Printable files of the curved fingertips (docs/handoff/20261015-curved_tips.md, steps 1 and 6). The tips print
upright: the mount face on the bed, the finger's axis vertical, 0.1 mm layers after a 0.2 mm first layer, so the palmar
surface's curve across the width is laid down as perimeters and only the distal cap is stepped. Runs in the femtip
environment (trimesh, manifold3d, rtree); PrusaSlicer 2.9.2 through flatpak slices the project.

Print frame (mm): x along the palmar normal (the palmar reach at x = 8.5), y across the finger (y_print = -y_tip), z up
from the mount face (z_print = 10.86 - z_tip; the distal end at 21.41), a proper rotation of the tip frame.

Magnet: tpu_tip_design's eFlesh disc (4.76 x 1.59 mm, N52, axially magnetized), axis along the palmar normal, its top
face 5 mm under the palmar reach at the contact point (x 1.91-3.50 mm, z 10.86 mm), as in the block tips. Printed
upright the disc stands on its edge in a slot: a seat of 90 % of the disc's diameter (tpu_tip_design.FIT_D, the owner's
press fit) closed by its lower half circle, walls of the same width running up to the pause, and the magnet's thickness
plus 0.1 mm across x. The print pauses before the first layer above the slot; the magnet is pressed down the slot to its
seat, north pole toward the palmar face, and the next layer bridges the 1.7 mm slot. Wall: the least TPU between the
slot and the outer surface: the nearest of 400,000 outer-surface samples (about 0.05 mm apart) to each of 20,000
samples of the slot's surface.

  stl    per tip: print/<design>/<date>-<design>_upright.stl and the wall check (print/walls.jsonl)
  print  per tip and recipe: <date>-<design>_<recipe>.3mf on the owner's v17 template (FLEX on extruder 2, 2 perimeters,
         gyroid at the recipe's density, 4 top and 4 bottom layers, the object's layer height 0.1 mm), the M601 pause,
         and <date>-<design>_<recipe>.json (print sheet, slice audit)

    ~/.claude/bin/resguard.sh run --mem 2G --cpu 100 -- taskset -c 19 ~/miniconda3/envs/femtip/bin/python \\
        scripts/curved_tips_print.py stl --tips tcd_cap6_u0.1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import curved_tips_common as C  # noqa: E402
import tip_codesign_geometry as TG  # noqa: E402
import tpu_tip_design as TD  # noqa: E402

OUT = C.D / "print"
WORK = C.LOGS / "print"
LAYER, FIRST = 0.1, 0.2
GAP = 0.10                      # air above the seated magnet, and across x, mm
RECIPES = {"w95_20": dict(grade="95A", fill="20%"), "w95_10": dict(grade="95A", fill="10%"),
           "w85_30": dict(grade="85A", fill="30%"), "w85_50": dict(grade="85A", fill="50%")}


def upright_body(design, res=0.0002):
    """The design body as a trimesh in the print frame (mm)."""
    import trimesh
    sh = TG.shape(design)
    V, F = sh.hull(res)
    P = np.column_stack([(V[:, 0] - (TG.REACH - 0.0085)) * 1e3, -V[:, 1] * 1e3, (TG.Z_MOUNT - V[:, 2]) * 1e3])
    m = trimesh.Trimesh(P, F, process=True)
    trimesh.repair.fix_normals(m)
    return m


def slot(magnet="d4.76x1.59", depth=5.0):
    """The magnet slot as a manifold (print frame, mm) and its dimensions."""
    import manifold3d as M
    D, t = TD.MAGNETS[magnet]
    r = TD.FIT_D * D / 2
    x_hi = TD.X_FACE - depth
    x_lo = x_hi - t - GAP
    zc = TG.Z_MOUNT * 1e3                                 # the contact point's height, 10.86 mm
    top = zc + D / 2 + (D / 2 - r) + GAP                  # the magnet sits (D/2 - r) above the seat's centre
    n_pause, z_pause, z_print = pause_layer(top)
    disc = M.Manifold.cylinder(x_hi - x_lo, r, r, 128).rotate([0.0, 90.0, 0.0]).translate([x_lo, 0.0, zc])
    box = M.Manifold.cube([x_hi - x_lo, 2 * r, z_pause - zc]).translate([x_lo, -r, zc])
    dims = dict(magnet=magnet, magnet_diameter_mm=D, magnet_thickness_mm=t, magnet_top_depth_mm=depth,
                seat_diameter_mm=round(2 * r, 3), slot_x_mm=[round(x_lo, 3), round(x_hi, 3)],
                slot_width_mm=round(2 * r, 3), seat_centre_z_mm=zc, slot_top_needed_mm=round(top, 3),
                slot_top_mm=z_pause, pause_before_layer=n_pause, pause_layer_bottom_mm=z_pause, pause_at_mm=z_print)
    return disc + box, dims


def pause_layer(z_top):
    """First layer whose bottom lies at or above z_top on the grid FIRST, FIRST + LAYER, ...: its number, its bottom and
    its top (PrusaSlicer's print_z, before which the custom G-code runs)."""
    k = max(0, math.ceil((z_top - FIRST) / LAYER - 1e-9))
    return k + 2, round(FIRST + k * LAYER, 4), round(FIRST + (k + 1) * LAYER, 4)


def build_part(design, magnet="d4.76x1.59", depth=5.0):
    import eflesh_build as EB
    import trimesh
    body = upright_body(design)
    cut, dims = slot(magnet, depth)
    part = EB.to_trimesh(EB.to_manifold(body) - cut)
    # wall: the nearest of 400,000 outer-surface samples to each of 20,000 slot-surface samples (a k-d tree; trimesh's
    # signed distance from points this deep in a fine mesh took 25 GB on 2026-10-15), and the slot inside the design
    from scipy.spatial import cKDTree
    sl = EB.to_trimesh(cut)
    pts = trimesh.sample.sample_surface(sl, 20000, seed=0)[0]
    outer = trimesh.sample.sample_surface(body, 400000, seed=1)[0]
    d, _ = cKDTree(outer).query(pts)
    tip = np.column_stack([pts[:, 0] * 1e-3 + (TG.REACH - 0.0085), -pts[:, 1] * 1e-3, TG.Z_MOUNT - pts[:, 2] * 1e-3])
    inside = TG.shape(design).inside_design(tip)
    dims.update(wall_min_mm=round(float(d.min()), 3), slot_inside_body=bool(inside.all()))
    return body, part, dims


def cmd_stl(a):
    rows = []
    for key in a.tips:
        design = key.rsplit("_", 1)[0] if ("_u" in key or "_p" in key) else key
        t0 = time.time()
        body, part, dims = build_part(design, a.magnet, a.depth)
        out = OUT / design
        out.mkdir(parents=True, exist_ok=True)
        stl = out / f"{C.DATE}-{design}_upright.stl"
        part.export(stl)
        row = dict(tip=key, design=design, stl=str(stl.relative_to(ROOT)), watertight=bool(part.is_watertight),
                   bounds_mm=np.round(part.bounds, 3).tolist(), body_volume_mm3=round(float(body.volume), 1),
                   part_volume_mm3=round(float(part.volume), 1), mount_face_area_mm2=round(mount_area(body), 1),
                   **dims, seconds=round(time.time() - t0, 1), script="scripts/curved_tips_print.py",
                   when=time.strftime("%Y-%m-%d %H:%M"))
        rows.append(row)
        with open(OUT / "walls.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(f"{design}: watertight {row['watertight']}, bounds {row['bounds_mm']}, part {row['part_volume_mm3']} mm3, "
              f"mount face {row['mount_face_area_mm2']} mm2, wall {row['wall_min_mm']} mm, pause before layer "
              f"{dims['pause_before_layer']} at {dims['pause_at_mm']} mm ({row['seconds']} s)", flush=True)
    return rows


def mount_area(body):
    """Area of the body's faces on the bed (z = 0), mm2."""
    F = body.faces
    z = body.vertices[F][:, :, 2]
    on = np.all(np.abs(z) < 1e-6, axis=1)
    return float(body.area_faces[on].sum())


def write_3mf(path, part, name, settings, pause_z, msg):
    """eflesh_build.write_3mf with the object's layer height set to LAYER in its model config."""
    import eflesh_build as EB
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    EB.write_3mf(tmp, part, name, EB.TEMPLATE, settings, 2, pause_z, msg, (180.0, 180.0))
    zin = zipfile.ZipFile(tmp)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for it in zin.infolist():
            data = zin.read(it.filename)
            if it.filename == "Metadata/Slic3r_PE_model.config":
                s = data.decode()
                anchor = '<metadata type="object" key="extruder" value="2"/>'
                assert anchor in s
                s = s.replace(anchor, anchor + f'\n  <metadata type="object" key="layer_height" value="{LAYER:g}"/>')
                data = s.encode()
            zout.writestr(it, data)
    zin.close()
    tmp.unlink()


def cmd_print(a):
    import eflesh_build as EB
    for key in a.tips:
        design = key.rsplit("_", 1)[0] if ("_u" in key or "_p" in key) else key
        body, part, dims = build_part(design, a.magnet, a.depth)
        out = OUT / design
        out.mkdir(parents=True, exist_ok=True)
        work = WORK / design
        work.mkdir(parents=True, exist_ok=True)
        stl = out / f"{C.DATE}-{design}_upright.stl"
        part.export(stl)
        _, _, kv = EB.template_config(EB.TEMPLATE)
        for rc in a.recipes:
            t0 = time.time()
            R = RECIPES[rc]
            settings = dict(perimeters=2, fill_density=R["fill"], fill_pattern="gyroid", top_solid_layers=4,
                            bottom_solid_layers=4)
            proj = out / f"{C.DATE}-{design}_{rc}.3mf"
            write_3mf(proj, part, f"{design}_{rc}", settings, dims["pause_at_mm"],
                      "Press the magnet down the slot, north pole toward the palmar face, and resume")
            rep = dict(tip=key, design=design, recipe=rc, **R, orientation="upright, mount face on the bed",
                       layer_height_mm=LAYER, first_layer_mm=FIRST, part_settings=settings, template=str(EB.TEMPLATE),
                       extruder=2, stl=str(stl.relative_to(ROOT)), project=str(proj.relative_to(ROOT)),
                       part_volume_mm3=round(float(part.volume), 1), watertight=bool(part.is_watertight),
                       bounds_mm=np.round(part.bounds, 3).tolist(), slot=dims,
                       note=("Print upright on the mount face. The slicer walls the slot with perimeters; pause before the "
                             f"layer at {dims['pause_at_mm']} mm, press the magnet down the slot to its seat with its north "
                             "pole toward the palmar face, and resume: the next layer bridges the slot."),
                       script="scripts/curved_tips_print.py", when=time.strftime("%Y-%m-%d %H:%M"))
            if not a.no_slice:
                row, gc = EB.slice_check(proj, work, kv)
                zs = sorted({float(l[3:]) for l in open(gc) if l.startswith(";Z:")})
                row["layer_heights_mm"] = sorted({round(b - a_, 3) for a_, b in zip(zs, zs[1:])})
                row["n_layers"] = len(zs)
                rep["slice"] = row
                rep["gcode"] = str(gc.relative_to(ROOT))
            rep["seconds"] = round(time.time() - t0, 1)
            (out / f"{C.DATE}-{design}_{rc}.json").write_text(json.dumps(rep, indent=1, default=float))
            sl = rep.get("slice", {})
            print(f"{design} {rc}: watertight {rep['watertight']}, bounds {rep['bounds_mm']}; pause {dims['pause_at_mm']} mm "
                  f"(slice: {sl.get('pause_found_before_layer_z')}); layers {sl.get('n_layers')} {sl.get('layer_heights_mm')}; "
                  + ", ".join(f"{q} {sl.get(q)}" for q in ("islands_max", "max_islands", "print_time", "time_min",
                                                          "supported_frac") if q in sl) + f" ({rep['seconds']} s)",
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("stl", "print"):
        p = sub.add_parser(name)
        p.add_argument("--tips", nargs="+", required=True)
        p.add_argument("--magnet", default="d4.76x1.59")
        p.add_argument("--depth", type=float, default=5.0)
    sub.choices["print"].add_argument("--recipes", nargs="+", default=["w95_20"], choices=list(RECIPES))
    sub.choices["print"].add_argument("--no-slice", action="store_true")
    a = ap.parse_args()
    {"stl": cmd_stl, "print": cmd_print}[a.cmd](a)


if __name__ == "__main__":
    main()
