#!/usr/bin/env python3
"""Convert the UR5 (CB3) visual meshes of ur_description (COLLADA .dae, metres, Z up) to one OBJ per material per link,
in the URDF's mesh frame, with the material's diffuse colour written to materials.json. MuJoCo reads OBJ and STL, not
DAE, so the visual meshes need this step; the collision meshes are STL already.

Runs under the conda base Python, the only interpreter here with pycollada:
    ~/miniconda3/bin/python logs/20261104-cad_v3/ur5/dae_to_obj.py <ur_description>/meshes/ur5/visual <out_dir>
"""
import json
import sys
from pathlib import Path

import collada
import numpy as np

STEMS = ("base", "shoulder", "upperarm", "forearm", "wrist1", "wrist2", "wrist3")


def load_groups(path):
    """{material name: (V (n, 3) m, F (m, 3), rgba)} of every triangle primitive in the DAE's scene, node transforms applied."""
    c = collada.Collada(str(path), ignore=[collada.common.DaeUnsupportedError, collada.common.DaeBrokenRefError])
    unit = float(getattr(c.assetInfo, "unitmeter", 1.0) or 1.0)
    groups = {}
    for bg in c.scene.objects("geometry"):
        for prim in bg.primitives():
            if not hasattr(prim, "vertex_index"):
                continue
            if type(prim).__name__ in ("BoundPolylist", "BoundPolygons"):
                prim = prim.triangleset()
            if type(prim).__name__ != "BoundTriangleSet":
                continue
            V = np.asarray(prim.vertex, float) * unit
            F = np.asarray(prim.vertex_index, int).reshape(-1, 3)
            mat = prim.material
            name = getattr(mat, "name", None) or getattr(mat, "id", None) or "none"
            diffuse = getattr(getattr(mat, "effect", None), "diffuse", None)
            rgba = list(diffuse) if isinstance(diffuse, (tuple, list)) else [0.7, 0.7, 0.7, 1.0]
            if len(rgba) == 3:
                rgba.append(1.0)
            g = groups.setdefault(name, [[], [], rgba])
            off = sum(len(v) for v in g[0])
            g[0].append(V)
            g[1].append(F + off)
    return {k: (np.concatenate(v[0]), np.concatenate(v[1]), v[2]) for k, v in groups.items()}, unit, c.assetInfo.upaxis


def write_obj(path, V, F):
    used = np.unique(F)                                   # drop vertices no face uses
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    V, F = V[used], remap[F]
    with open(path, "w") as fh:
        fh.write("# UR5 (CB3) visual mesh from ur_description (ros-industrial/universal_robot), metres, URDF mesh frame\n")
        fh.write("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in V))
        fh.write("".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in F))
    return len(V), len(F)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    info = {}
    for stem in STEMS:
        groups, unit, up = load_groups(src / f"{stem}.dae")
        rows = []
        # order: largest group first (so <stem>_0 is the body colour)
        for k, (name, (V, F, rgba)) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1][1]))):
            fn = f"{stem}_{k}.obj"
            nv, nf = write_obj(out / fn, V, F)
            rows.append(dict(file=fn, material=name, rgba=[round(float(x), 4) for x in rgba], n_vertices=nv, n_faces=nf,
                             lo=np.round(V[np.unique(F)].min(0), 5).tolist(), hi=np.round(V[np.unique(F)].max(0), 5).tolist()))
        info[stem] = dict(unit_m=unit, up_axis=str(up), parts=rows)
        print(stem, unit, up, [(r["file"], r["material"], r["rgba"], r["n_faces"]) for r in rows])
    (out / "materials.json").write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
