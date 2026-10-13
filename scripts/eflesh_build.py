#!/usr/bin/env python3
"""An eFlesh lattice part from a closed shape in one command, ready for PrusaSlicer (2026-10-13).

Owner, 2026-10-13: "last i worked with it, i had to create a cad file, run it through a jupyter notebook, then blender to
slice in the magnets, and then back into a cad software to repair the file so that it could be printed ... surely there
are improvements we can make now".

Steps, all in this script:
  1. shape: an STL or OBJ in print orientation (z up, any closed surface), or `sr2tip`, the whole SR2 fingertip printed
     face up (scripts/tpu_tip_design.py);
  2. lattice: eFlesh's stitch tool fills the shape's bounding box with whole cells of pattern 0646 at the requested
     modulus (its cut-cell tool crashes in OpenVDB on every input on this workstation, 2026-10-13);
  3. cut: exact booleans in manifold3d (no Blender, no repair) keep the lattice inside the shape, open on its faces as eFlesh
     prints it, on a solid base of --base mm on the bed; --skin adds a closed skin (every face, or the top face only), which
     eFlesh does not have;
  4. pouches: eFlesh's pouch (a cup with a 1 mm wall and floor and a lip whose hole is 0.8 of the cavity's radius) at
     each magnet, axis along z, the lattice cleared around it;
  5. project: a PrusaSlicer 3MF with the owner's printer, filament and print settings copied from a template project
     (default: the printed v17), the part's own settings, and a pause before the first layer above the highest cavity;
  6. check: PrusaSlicer slices the project; the report lists the pause found in the G-code, the islands per layer, the
     share of each layer's beads that lies on the layer below (scripts/eflesh_printability.py) and the print time.

    ~/miniconda3/envs/femtip/bin/python scripts/eflesh_build.py --shape sr2tip --cell 8 --E 0.0035 \
        --magnet 4.76,1.59,c,c,5 --base 1.6 --name sr2tip_c8_E0035_open
Magnet: diameter,thickness,x,y-or-c,depth: x and y in the shape's frame (c: under the contact point of sr2tip), depth
from the shape's top to the magnet's top face.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import trimesh
import manifold3d as M

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_tip_design as TD  # noqa: E402

EFLESH = Path.home() / "wxie_workspace/eFlesh/microstructure"
STITCH = EFLESH / "microstructure_inflators/build/isosurface_inflator/stitch_cells_cli"
PATTERN = EFLESH / "microstructure_inflators/data/patterns/3D/reference_wires/pattern0646.wire"
COEFFS = EFLESH / "matopt/tools/material2geometry/0646_geo_1_coeffs.txt"
TEMPLATE = Path.home() / "wxie_workspace/fingers/merged_finger_v17.3mf"
OUT = ROOT / "docs/experiments/20261013-eflesh_printability/build"
WORK = ROOT / "logs/20261013-eflesh_build"
PRUSA = ["flatpak", "run", "--command=prusa-slicer", "com.prusa3d.PrusaSlicer"]


def to_manifold(tm: trimesh.Trimesh) -> M.Manifold:
    m = M.Manifold(M.Mesh(vert_properties=np.asarray(tm.vertices, np.float32), tri_verts=np.asarray(tm.faces, np.uint32)))
    if m.status() != M.Error.NoError:
        raise SystemExit(f"shape is not a closed manifold: {m.status()}")
    return m


def to_trimesh(m: M.Manifold) -> trimesh.Trimesh:
    mm = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)


def shape(spec):
    """Shape in print orientation with its lowest point on z = 0, and the contact point (x, y) if it has one."""
    if spec == "sr2tip":
        m = TD.mesh(None, None)
        R = np.eye(4); R[:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]      # palmar normal x -> print z, face up
        m.apply_transform(R)
        shift = -m.bounds[0]
        m.apply_translation(shift)
        contact = (-TD.Z_C + shift[0], 0.0 + shift[1])
        return m, contact
    m = trimesh.load(spec, force="mesh")
    m.apply_translation([0, 0, -m.bounds[0][2]])
    return m, None


def material_params(E, nu):
    sys.path.insert(0, str(COEFFS.parent))
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        from material2geometry import Material2Geometry
        return Material2Geometry(in_path=str(COEFFS)).evaluate(nu, E)


def lattice(lo, hi, cell, E, nu, origin, work):
    """eFlesh's stitch tool over whole cells covering [lo, hi], shifted so that a cell corner sits at `origin`."""
    p = material_params(E, nu)
    i0 = np.floor((lo - origin) / cell).astype(int)
    i1 = np.ceil((hi - origin) / cell).astype(int) - 1
    cells = [dict(params=p, symmetry="Cubic", pattern=str(PATTERN), index=[i, j, k])
             for i in range(i0[0], i1[0] + 1) for j in range(i0[1], i1[1] + 1) for k in range(i0[2], i1[2] + 1)]
    work.mkdir(parents=True, exist_ok=True)
    pj, obj = work / "patch.json", work / "lattice.obj"
    json.dump(cells, open(pj, "w"))
    t = time.time()
    r = subprocess.run([str(STITCH), "-p", str(pj), "--gridSize", str(cell), "-o", str(obj), "-r", "50"], capture_output=True, text=True)
    if r.returncode != 0 or not obj.exists():
        raise SystemExit(f"stitch tool failed: {r.stderr[-500:]}")
    lat = trimesh.load(obj, force="mesh", process=False)
    lat.apply_translation(origin)
    thick = np.array(p[5:9]) * cell
    return to_manifold(lat), dict(cells=len(cells), stitch_s=round(time.time() - t, 1), strut_mm=thick.round(3).tolist(),
                                  faces_raw=int(len(lat.faces)))


def pouch(d, t, x, y, top, wall=1.0, floor=1.0, lip=1.0, clear_r=0.15, clear_t=0.10, hole=0.8, seg=96):
    """eFlesh's pouch around an axially magnetized disc whose top face is at height `top`."""
    rc, lo, hi = d / 2 + clear_r, top - t - clear_t / 2, top + clear_t / 2
    def cyl(r, z0, z1):
        return M.Manifold.cylinder(z1 - z0, r, r, seg).translate((x, y, z0))
    outer = cyl(rc + wall, lo - floor, hi + lip)
    cup = outer - cyl(rc, lo, hi) - cyl(hole * rc, hi - 0.01, hi + lip + 1.0)
    return cup, outer, dict(d=d, t=t, x=x, y=y, magnet_top=top, cavity=[round(lo, 3), round(hi, 3)], cavity_r=round(rc, 3),
                            hole_r=round(hole * rc, 3), outer_r=round(rc + wall, 3), outer_z=[round(lo - floor, 3), round(hi + lip, 3)])


def pause_layer(z_cavity_top, layer, first):
    """First layer whose mid-height lies above the cavity's top: (its number, its print_z)."""
    n = max(1, int(math.ceil((z_cavity_top - first) / layer + 0.5 - 1e-9)) + 1) if z_cavity_top > first else 1
    return n, round(first + (n - 1) * layer, 4)


def template_config(path):
    z = zipfile.ZipFile(path)
    cfg = z.read("Metadata/Slic3r_PE.config").decode()
    kv = {}
    for line in cfg.splitlines():
        if line.startswith("; ") and " = " in line:
            k, v = line[2:].split(" = ", 1)
            kv[k] = v
    return z, cfg, kv


def write_3mf(path, part: trimesh.Trimesh, name, template, settings, extruder, pause_z, pause_msg, bed_xy):
    """PrusaSlicer project: one object, one part, the template's configuration, a pause at pause_z."""
    z, cfg, _ = template_config(template)
    V, F = np.asarray(part.vertices), np.asarray(part.faces)
    verts = "\n".join(f'     <vertex x="{a:.5f}" y="{b:.5f}" z="{c:.5f}"/>' for a, b, c in V)
    tris = "\n".join(f'     <triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in F)
    model = (f'<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
             f'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">\n'
             f' <metadata name="slic3rpe:Version3mf">1</metadata>\n <metadata name="Title">{name}</metadata>\n'
             f' <metadata name="Application">scripts/eflesh_build.py</metadata>\n <resources>\n  <object id="1" type="model">\n   <mesh>\n'
             f'    <vertices>\n{verts}\n    </vertices>\n    <triangles>\n{tris}\n    </triangles>\n   </mesh>\n  </object>\n </resources>\n'
             f' <build>\n  <item objectid="1" transform="1 0 0 0 1 0 0 0 1 {bed_xy[0]:.4f} {bed_xy[1]:.4f} 0" printable="1"/>\n </build>\n</model>\n')
    vol = "".join(f'   <metadata type="volume" key="{k}" value="{v}"/>\n' for k, v in settings.items())
    mcfg = (f'<?xml version="1.0" encoding="UTF-8"?>\n<config>\n <object id="1" instances_count="1">\n'
            f'  <metadata type="object" key="name" value="{name}"/>\n  <metadata type="object" key="extruder" value="{extruder}"/>\n'
            f'  <volume firstid="0" lastid="{len(F) - 1}">\n   <metadata type="volume" key="name" value="{name}"/>\n'
            f'   <metadata type="volume" key="volume_type" value="ModelPart"/>\n'
            f'   <metadata type="volume" key="matrix" value="1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"/>\n{vol}'
            f'   <mesh edges_fixed="0" degenerate_facets="0" facets_removed="0" facets_reversed="0" backwards_edges="0"/>\n'
            f'  </volume>\n </object>\n</config>\n')
    pz = ""
    if pause_z is not None:
        pz = (f'<?xml version="1.0" encoding="utf-8"?>\n<custom_gcodes_per_print_z bed_idx="0">\n'
              f'<code print_z="{pause_z:g}" type="1" extruder="{extruder}" color="" extra="{pause_msg}" gcode="M601"/>\n'
              f'<mode value="MultiAsSingle"/>\n</custom_gcodes_per_print_z>\n')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as out:
        out.writestr("[Content_Types].xml", z.read("[Content_Types].xml"))
        out.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
                     ' <Relationship Target="/3D/3dmodel.model" Id="rel-1" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n</Relationships>\n')
        out.writestr("3D/3dmodel.model", model)
        out.writestr("Metadata/Slic3r_PE.config", cfg)
        out.writestr("Metadata/Slic3r_PE_model.config", mcfg)
        if pz:
            out.writestr("Metadata/Prusa_Slicer_custom_gcode_per_print_z.xml", pz)


def slice_check(project, work, kv):
    """Slice the project with PrusaSlicer, as the owner would, and read back the pause and the toolpaths."""
    ini = work / "template.ini"
    ini.write_text("\n".join(f"{k} = {v}" for k, v in kv.items() if k != "binary_gcode") + "\nbinary_gcode = 0\n")
    gc = work / (project.stem + ".gcode")
    r = subprocess.run(PRUSA + ["--load", str(ini), "--export-gcode", "-o", str(gc), str(project)], capture_output=True, text=True)
    if r.returncode != 0 or not gc.exists():
        raise SystemExit(f"PrusaSlicer failed: {(r.stdout + r.stderr)[-800:]}")
    pause_at, z = None, None
    for line in open(gc):
        if line.startswith(";Z:"):
            z = float(line[3:])
        elif line.startswith("M601") and pause_at is None:
            pause_at = z
    import eflesh_printability as EP
    row = EP.audit(project.stem, gc, "built by scripts/eflesh_build.py", (0, 0))
    row.pop("per_layer", None)
    row["pause_found_before_layer_z"] = pause_at   # PrusaSlicer writes M601 after the ;Z: of the layer it holds back
    return row, gc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", required=True, help="sr2tip, or an STL/OBJ in print orientation")
    ap.add_argument("--cell", type=float, default=8.0, help="cell size, mm (8 prints on a 0.4 mm nozzle; 6 needs 0.25)")
    ap.add_argument("--E", type=float, default=0.0035, help="eFlesh's target modulus over the solid's")
    ap.add_argument("--nu", type=float, default=0.09)
    ap.add_argument("--origin", default="contact", help="x,y,z of a cell corner, or 'contact' (a corner under the contact point)")
    ap.add_argument("--magnet", action="append", default=[], help="diameter,thickness,x,y,depth (x or y may be c: contact point)")
    ap.add_argument("--skin", type=float, default=0.0,
                    help="skin thickness, mm; 0 (default) leaves the lattice open as eFlesh prints it")
    ap.add_argument("--skin-mode", choices=["all", "top", "gap"], default="all",
                    help="all: every face; top: the top face only; gap: the top face, and walls that rise from the base and stop "
                         "--wall-gap mm below the top skin, so that they close the sides without carrying the contact to the base")
    ap.add_argument("--wall-gap", type=float, default=1.5, help="height of the open band between the walls and the top skin, mm")
    ap.add_argument("--base", type=float, default=1.6, help="solid base on the bed, mm")
    ap.add_argument("--template", default=str(TEMPLATE), help="PrusaSlicer 3MF whose settings the project takes")
    ap.add_argument("--extruder", type=int, default=2, help="tool of the part (the v17 project used tool 2, FLEX)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-slice", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    work = WORK / a.name
    work.mkdir(parents=True, exist_ok=True)
    tm, contact = shape(a.shape)
    S = to_manifold(tm)
    lo, hi = np.asarray(tm.bounds[0]), np.asarray(tm.bounds[1])
    if a.origin == "contact":
        cxy = contact if contact is not None else tuple(0.5 * (lo[:2] + hi[:2]))
        origin = np.array([cxy[0] - a.cell / 2, cxy[1] - a.cell / 2, a.base])
    else:
        origin = np.array([float(v) for v in a.origin.split(",")])
    L, linfo = lattice(lo, hi, a.cell, a.E, a.nu, origin, work)

    if a.skin > 0 and a.skin_mode == "all":
        inner = S.minkowski_difference(M.Manifold.sphere(a.skin, 48))
    elif a.skin > 0 and a.skin_mode == "top":
        inner = S ^ S.translate((0.0, 0.0, -a.skin))            # below a cap of the skin's thickness under the top face
    elif a.skin > 0:                                             # gap: top cap, and walls from the base to wall_gap below it
        cap = S - S.translate((0.0, 0.0, -a.skin))
        walls = S - S.minkowski_difference(M.Manifold.sphere(a.skin, 48))
        zc = float(cap.bounding_box()[2]) - a.wall_gap
        below = M.Manifold.cube((hi[0] - lo[0] + 2, hi[1] - lo[1] + 2, zc)).translate((lo[0] - 1, lo[1] - 1, 0.0))
        inner = S - (cap + (walls ^ below))
    else:
        inner = S
    slab = M.Manifold.cube((hi[0] - lo[0] + 2, hi[1] - lo[1] + 2, a.base)).translate((lo[0] - 1, lo[1] - 1, 0.0))
    base = S ^ slab if a.base > 0 else None
    core = (L ^ inner)
    if base is not None:
        core = core - slab
    shell = (S - inner) if a.skin > 0 else None

    pouches, info_p = [], []
    for spec in a.magnet:
        d, t, x, y, depth = spec.split(",")
        x = contact[0] if x == "c" else float(x)
        y = contact[1] if y == "c" else float(y)
        cup, hull, info = pouch(float(d), float(t), x, y, hi[2] - float(depth))
        core = core - hull
        pouches.append(cup)
        info_p.append(info)
    part = core
    for extra in [shell, base] + pouches:
        if extra is not None:
            part = part + extra
    part = part ^ S                                         # nothing outside the shape
    pm = to_trimesh(part)
    _, _, kv = template_config(a.template)
    layer, first = float(kv["layer_height"]), float(kv["first_layer_height"])
    pause = pause_layer(max(p["cavity"][1] for p in info_p), layer, first) if info_p else (None, None)
    settings = dict(perimeters=2, fill_density="0%", top_solid_layers=max(3, int(math.ceil(a.skin / layer))) if a.skin > 0 else 5,
                    bottom_solid_layers=max(3, int(math.ceil(a.base / layer))))
    OUT.mkdir(parents=True, exist_ok=True)
    proj = OUT / f"20261013-{a.name}.3mf"
    write_3mf(proj, pm, a.name, a.template, settings, a.extruder, pause[1],
              "Press the magnets into the pouches, north pole up, and resume", (180.0, 180.0))
    pm.export(work / f"{a.name}.stl")
    rep = dict(name=a.name, shape=a.shape, cell_mm=a.cell, E=a.E, nu=a.nu, origin=origin.round(3).tolist(), skin_mm=a.skin,
               skin_mode=a.skin_mode, wall_gap_mm=a.wall_gap if a.skin_mode == "gap" else None,
               base_mm=a.base, lattice=linfo, pouches=info_p, part_faces=int(len(pm.faces)), part_volume_mm3=round(float(pm.volume), 1),
               shape_volume_mm3=round(float(tm.volume), 1), solid_fraction=round(float(pm.volume / tm.volume), 4),
               watertight=bool(pm.is_watertight), project=str(proj.relative_to(ROOT)),
               project_MB=round(proj.stat().st_size / 1e6, 2), pause_layer=pause[0], pause_print_z=pause[1],
               part_settings=settings, template=a.template, extruder=a.extruder)
    if not a.no_slice:
        row, gc = slice_check(proj, work, kv)
        rep["slice"] = row
    rep["seconds"] = round(time.time() - t0, 1)
    json.dump(rep, open(OUT / f"20261013-{a.name}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("lattice", "pouches")}, indent=1))


if __name__ == "__main__":
    main()
