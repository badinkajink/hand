#!/usr/bin/env python3
"""Printable files of the fingertip co-design finalists (docs/notes/20261014-tip_codesign_plan.md, step 11). Runs in the
femtip environment (trimesh, manifold3d); PrusaSlicer 2.9.2 through flatpak slices the project.

Per tip key (an as-printed key prints its design surface; the slicer makes the terraces), under
docs/experiments/20261014-tip_codesign/print/<design key>/:

  20261014-<key>_faceup.stl          the body palmar face up on its back face, mm: x along the finger from the mount face,
                                     y across, z up from the bed (17.0 mm at the apex); the press-fit magnet pocket is an
                                     inner void (tpu_tip_design.pocket: 90 % of the magnet's diameter, floor and top on the
                                     0.2 mm layer grid, its axis under the contact point)
  20261014-<key>_<recipe>.3mf        PrusaSlicer project on the owner's v17 template: FLEX on extruder 2, 2 perimeters,
                                     gyroid at the recipe's density, 4 top and 4 bottom layers, an M601 pause before the
                                     first layer above the pocket (press the magnet in, north pole up, and resume)
  20261014-<key>_<recipe>.json       print sheet (tpu_tip_design.print_sheet), recipe, part settings, the slice audit
                                     (pause height, islands per layer, supported share, print time)

    ~/.claude/bin/resguard.sh run --mem 2G --cpu 100 -- taskset -c 4 ~/miniconda3/envs/femtip/bin/python \\
        scripts/tip_codesign_print.py --tips tcd_rs3_re3_p0.2 --recipes w95_20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eflesh_build as EB  # noqa: E402
import tip_codesign_geometry as TG  # noqa: E402
import tpu_tip_design as TD  # noqa: E402

OUT = ROOT / "docs/experiments/20261014-tip_codesign/print"
WORK = ROOT / "logs/20261014-tip_codesign/print"
DATE = "20261014"
RECIPES = {   # tpu_tip_magnet.RECIPES: grade, gyroid density; E* of the whole tip with the 4.76 mm magnet 5 mm deep (FE)
    "w95_20": dict(grade="95A", fill="20%", E_star_MPa=14.6),
    "w95_10": dict(grade="95A", fill="10%", E_star_MPa=7.2),
    "w85_30": dict(grade="85A", fill="30%", E_star_MPa=6.9),
    "w85_50": dict(grade="85A", fill="50%", E_star_MPa=16.5),
}


def faceup_body(key, res=0.00025):
    """The design body as a trimesh in print orientation (mm): (x, y, z)_print = (z_ins, -y_ins, x_ins + 8.5), a proper
    rotation of the insert frame (palmar +x at 8.5 mm, mount face z = 0)."""
    import trimesh
    sh = TG.shape(key)
    V, F = sh.hull(res)
    ins = np.column_stack([(V[:, 0] - (TG.REACH - 0.0085)) * 1e3, V[:, 1] * 1e3, (TG.Z_MOUNT - V[:, 2]) * 1e3])
    P = np.column_stack([ins[:, 2], -ins[:, 1], ins[:, 0] + 8.5])
    m = trimesh.Trimesh(P, F, process=True)
    trimesh.repair.fix_normals(m)
    return m


def pocket_cylinder(magnet, depth, seg=96):
    import manifold3d as M
    pk, mg = TD.pocket(magnet, depth)
    z0, z1 = pk["x_lo"] + 8.5, pk["x_hi"] + 8.5
    cyl = M.Manifold.cylinder(z1 - z0, pk["r"], pk["r"], seg).translate((TD.Z_C, 0.0, z0))
    return cyl, pk, mg


def build(key, recipe, magnet="d4.76x1.59", depth=5.0, slice_=True):
    design = key.rsplit("_p", 1)[0] if "_p" in key else key
    out = OUT / design
    out.mkdir(parents=True, exist_ok=True)
    work = WORK / design
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    body = faceup_body(design)
    S = EB.to_manifold(body)
    cyl, pk, mg = pocket_cylinder(magnet, depth)
    part = EB.to_trimesh(S - cyl)
    stl = out / f"{DATE}-{design}_faceup.stl"
    part.export(stl)
    sheet = TD.print_sheet(magnet, depth)
    _, _, kv = EB.template_config(EB.TEMPLATE)
    R = RECIPES[recipe]
    settings = dict(perimeters=2, fill_density=R["fill"], fill_pattern="gyroid", top_solid_layers=4, bottom_solid_layers=4)
    proj = out / f"{DATE}-{design}_{recipe}.3mf"
    EB.write_3mf(proj, part, f"{design}_{recipe}", EB.TEMPLATE, settings, 2, sheet["pause_before_layer_at_mm"],
                 "Press the magnet into the pocket, north pole up, and resume", (180.0, 180.0))
    rep = dict(tip=key, design=design, recipe=recipe, **R, magnet=magnet, magnet_depth_mm=depth, print_sheet=sheet,
               part_settings=settings, template=str(EB.TEMPLATE), extruder=2, stl=str(stl.relative_to(ROOT)),
               project=str(proj.relative_to(ROOT)), part_volume_mm3=round(float(part.volume), 1),
               body_volume_mm3=round(float(body.volume), 1), watertight=bool(part.is_watertight),
               bounds_mm=np.round(part.bounds, 3).tolist(), pocket=pk, magnet_cylinder=mg,
               script="scripts/tip_codesign_print.py", when=time.strftime("%Y-%m-%d %H:%M"))
    if slice_:
        row, gc = EB.slice_check(proj, work, kv)
        rep["slice"] = row
        rep["gcode"] = str(gc.relative_to(ROOT))
    rep["seconds"] = round(time.time() - t0, 1)
    (out / f"{DATE}-{design}_{recipe}.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tips", nargs="+", required=True)
    ap.add_argument("--recipes", nargs="+", default=["w95_20"], choices=list(RECIPES))
    ap.add_argument("--magnet", default="d4.76x1.59")
    ap.add_argument("--depth", type=float, default=5.0)
    ap.add_argument("--no-slice", action="store_true")
    a = ap.parse_args()
    for k in a.tips:
        for rc in a.recipes:
            r = build(k, rc, a.magnet, a.depth, not a.no_slice)
            sl = r.get("slice", {})
            print(f"{r['design']} {rc}: watertight {r['watertight']}, bounds {r['bounds_mm']}, part {r['part_volume_mm3']} mm3; "
                  f"pause {r['print_sheet']['pause_before_layer_at_mm']} mm (slice: {sl.get('pause_found_before_layer_z')}); "
                  + ", ".join(f"{q} {sl.get(q)}" for q in ("max_islands", "islands_max", "time_min", "print_time", "supported_frac")
                              if q in sl) + f" ({r['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
