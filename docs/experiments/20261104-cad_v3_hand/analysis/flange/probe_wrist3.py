"""Probe of the UR5 (CB3) wrist 3 meshes of the lab's ur_description: where the flange face is, which features of the
tool flange (M6 holes, dowel hole, recess, tool connector) the meshes carry, and where they sit in tool0 (E).

The visual mesh is COLLADA (no pycollada in the venv), so it is read here with ElementTree. The URDF places both meshes
in wrist_3_link at origin xyz (0, 0, -0.0818) rpy (pi/2, 0, 0); tool0 = wrist_3_link (flange -> tool0 composed, checked
below).

    ~/.claude/bin/resguard.sh run --mem 3G -- .venv/bin/python logs/20261104-cad_v3/flange/probe_wrist3.py
"""
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

UR = Path.home() / "Programs/rh56_controller/magpie_force_control/src/ur_description"
OUT = Path(__file__).resolve().parent
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def load_dae(path):
    root = ET.parse(path).getroot()
    geo = root.find(".//c:library_geometries/c:geometry", NS)
    pos = None
    for src in geo.findall(".//c:source", NS):
        if src.get("id").endswith("positions"):
            pos = np.array(src.find("c:float_array", NS).text.split(), float).reshape(-1, 3)
    groups = []
    for tri in geo.findall(".//c:triangles", NS):
        inputs = tri.findall("c:input", NS)
        stride = max(int(i.get("offset")) for i in inputs) + 1
        voff = [int(i.get("offset")) for i in inputs if i.get("semantic") == "VERTEX"][0]
        p = np.array(tri.find("c:p", NS).text.split(), int).reshape(-1, stride)[:, voff].reshape(-1, 3)
        groups.append((tri.get("material"), p))
    node = root.find(".//c:library_visual_scenes//c:node[c:instance_geometry]", NS)
    M = np.array(node.find("c:matrix", NS).text.split(), float).reshape(4, 4)
    return pos, groups, M


def rpy(r, p, y):
    return Rotation.from_euler("xyz", [r, p, y]).as_matrix()   # extrinsic xyz = Rz Ry Rx


def main():
    # tool0 in wrist_3_link from the URDF's flange and flange-tool0 joints
    R_wf = rpy(0, -math.pi / 2, -math.pi / 2)
    R_ft = rpy(math.pi / 2, 0, math.pi / 2)
    R_wt = R_wf @ R_ft
    print("R wrist_3_link <- tool0 =\n", np.round(R_wt, 9))
    # mesh -> wrist_3_link (= tool0), mm
    T_wm = np.eye(4)
    T_wm[:3, :3] = rpy(math.pi / 2, 0, 0)
    T_wm[:3, 3] = [0, 0, -81.8]
    pos, groups, M = load_dae(UR / "meshes/ur5/visual/wrist3.dae")
    print("node matrix\n", M)
    V = (pos * 1e3) @ M[:3, :3].T + M[:3, 3] * 1e3
    VE = V @ T_wm[:3, :3].T + T_wm[:3, 3]
    faces = np.vstack([g[1] for g in groups])
    mats = np.concatenate([[g[0]] * len(g[1]) for g in groups])
    mesh = trimesh.Trimesh(VE, faces, process=False)
    print("visual mesh in E: verts", len(VE), "faces", len(faces), "bounds", np.round(mesh.bounds, 2).tolist())
    for g in groups:
        vv = VE[np.unique(g[1])]
        print(" material", g[0], "faces", len(g[1]), "bounds", np.round(vv.min(0), 2).tolist(), np.round(vv.max(0), 2).tolist())
    col = trimesh.load(UR / "meshes/ur5/collision/wrist3.stl", force="mesh")
    col.apply_scale(1e3)
    col.apply_transform(T_wm)
    print("collision mesh in E: verts", len(col.vertices), "faces", len(col.faces), "bounds", np.round(col.bounds, 2).tolist())
    # z levels with many vertices (planar faces normal to the tool axis)
    zs, cnt = np.unique(np.round(VE[:, 2], 2), return_counts=True)
    order = np.argsort(-cnt)[:25]
    print("most populated z levels (E, mm):", [(float(zs[i]), int(cnt[i])) for i in sorted(order, key=lambda i: zs[i])])
    # vertices on the flange face and just behind it, in polar coordinates about the tool axis
    for zlo, zhi in ((-0.6, 0.6), (-8.0, -0.6), (-20, -8)):
        sel = (VE[:, 2] > zlo) & (VE[:, 2] <= zhi)
        r = np.hypot(VE[sel, 0], VE[sel, 1])
        a = np.degrees(np.arctan2(VE[sel, 1], VE[sel, 0]))
        rr, rc = np.unique(np.round(r, 1), return_counts=True)
        print(f"z in ({zlo},{zhi}]: {sel.sum()} verts; radii with >= 6 verts:",
              [(float(x), int(c)) for x, c in zip(rr, rc) if c >= 6])
    np.save(OUT / "wrist3_visual_E.npy", VE)
    json.dump(dict(faces=faces.tolist(), materials=mats.tolist()), open(OUT / "wrist3_visual_faces.json", "w"))
    col.export(OUT / "wrist3_collision_E.stl")


if __name__ == "__main__":
    main()
