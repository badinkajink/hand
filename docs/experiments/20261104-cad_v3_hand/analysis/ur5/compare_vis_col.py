#!/usr/bin/env python3
"""Do the converted visual OBJs sit where the collision STLs sit (both in the URDF's mesh frame)? Bounds per link and the
distance from 3000 surface samples of each to the other (mm).

    ~/.claude/bin/resguard.sh run --mem 3G -- .venv/bin/python logs/20261104-cad_v3/ur5/compare_vis_col.py
"""
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
UR = Path.home() / "Programs/rh56_controller/magpie_force_control/src/ur_description/meshes/ur5"
info = json.loads((HERE / "meshes/visual/materials.json").read_text())
out = {}
for stem, v in info.items():
    vis = trimesh.util.concatenate([trimesh.load(HERE / "meshes/visual" / p["file"], force="mesh", process=False)
                                    for p in v["parts"]])
    col = trimesh.load(UR / "collision" / f"{stem}.stl", force="mesh")
    pv, _ = trimesh.sample.sample_surface(vis, 3000, seed=0)
    pc, _ = trimesh.sample.sample_surface(col, 3000, seed=0)
    dense_v, _ = trimesh.sample.sample_surface(vis, 60000, seed=1)
    dense_c, _ = trimesh.sample.sample_surface(col, 60000, seed=1)
    d_vc = cKDTree(dense_c).query(pv)[0] * 1e3          # visual sample -> nearest collision sample (mm)
    d_cv = cKDTree(dense_v).query(pc)[0] * 1e3
    row = dict(vis_lo_mm=np.round(vis.bounds[0] * 1e3, 1).tolist(), vis_hi_mm=np.round(vis.bounds[1] * 1e3, 1).tolist(),
               col_lo_mm=np.round(col.bounds[0] * 1e3, 1).tolist(), col_hi_mm=np.round(col.bounds[1] * 1e3, 1).tolist(),
               vis_faces=int(len(vis.faces)), col_faces=int(len(col.faces)), col_watertight=bool(col.is_watertight),
               col_volume_cm3=round(float(col.volume) * 1e6, 1),
               vis_to_col_mm_p50_p95_max=np.round(np.percentile(d_vc, [50, 95, 100]), 2).tolist(),
               col_to_vis_mm_p50_p95_max=np.round(np.percentile(d_cv, [50, 95, 100]), 2).tolist())
    out[stem] = row
    print(stem, json.dumps(row))
(HERE / "compare_vis_col.json").write_text(json.dumps(out, indent=1))
