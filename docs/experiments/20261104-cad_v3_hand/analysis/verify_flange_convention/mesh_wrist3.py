"""Independent read of the UR5 (CB3) wrist-3 visual mesh of the lab's ur_description, placed in tool0 (E) by the URDF:
which tool-flange features it models (M6 holes, 6 mm pin hole, recess, tool connector) and at what angle about z_E.

Written separately from logs/20261104-cad_v3/flange/probe_wrist3.py: plain numpy on the COLLADA arrays, the placement
taken from the URDF text (check_frames.json: mesh +x -> +x_E, +y -> +z_E, +z -> -y_E, then z_E - 81.8 mm).

    ~/.claude/bin/resguard.sh run --mem 1G -- .venv/bin/python -I logs/20261104-cad_v3/verify_flange_convention/mesh_wrist3.py
"""
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
UR = Path.home() / "Programs/rh56_controller/magpie_force_control/src/ur_description"
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def read_dae(path):
    root = ET.parse(path).getroot()
    unit = float(root.find(".//c:asset/c:unit", NS).get("meter"))
    up = root.find(".//c:asset/c:up_axis", NS).text.strip()
    arrs = {fa.get("id"): np.array(fa.text.split(), float) for fa in root.iter("{%s}float_array" % NS["c"])}
    pos = [v for k, v in arrs.items() if k.endswith("positions-array")][0].reshape(-1, 3)
    tris = []
    for tri in root.iter("{%s}triangles" % NS["c"]):
        ins = tri.findall("c:input", NS)
        stride = max(int(i.get("offset")) for i in ins) + 1
        vo = [int(i.get("offset")) for i in ins if i.get("semantic") == "VERTEX"][0]
        tris.append(np.array(tri.find("c:p", NS).text.split(), int).reshape(-1, stride)[:, vo].reshape(-1, 3))
    node = [n for n in root.iter("{%s}node" % NS["c"]) if n.find("c:instance_geometry", NS) is not None][0]
    M = np.array(node.find("c:matrix", NS).text.split(), float).reshape(4, 4)
    return pos * unit, np.vstack(tris), M, up


def circle_fit(xy):
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = math.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)
    res = np.hypot(*(xy - c[:2]).T) - r
    return c[:2], r, float(np.sqrt((res ** 2).mean()))


def main():
    P, F, M, up = read_dae(UR / "meshes/ur5/visual/wrist3.dae")
    assert np.allclose(M, np.eye(4)) and up == "Z_UP", (M, up)
    P = P * 1e3                                                     # mm, mesh frame
    E = np.c_[P[:, 0], -P[:, 2], P[:, 1] - 81.8]                    # tool0: Rx(+90 deg) then z - 81.8 mm
    rho = np.hypot(E[:, 0], E[:, 1])
    ang = np.degrees(np.arctan2(E[:, 1], E[:, 0]))
    out = dict(n_vertices=len(E), n_faces=len(F), z_range_E_mm=[float(E[:, 2].min()), float(E[:, 2].max())],
               rho_max_mm=float(rho.max()))
    # flange face: the highest z level that many vertices share
    zs, cnt = np.unique(np.round(E[:, 2], 2), return_counts=True)
    top = zs[cnt >= 20].max()
    out["face_z_E_mm"] = float(top)
    # holes: vertices within 8.5 mm below the face, on 18..30 mm radius, grouped by angle
    sel = (E[:, 2] < top - 0.3) & (E[:, 2] > top - 8.5) & (rho > 18) & (rho < 30)
    a = np.sort(ang[sel])
    holes = []
    if sel.any():
        pts = E[sel]
        order = np.argsort(np.degrees(np.arctan2(pts[:, 1], pts[:, 0])))
        pts = pts[order]
        aa = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
        gaps = np.where(np.diff(aa) > 6.0)[0]
        groups = np.split(np.arange(len(pts)), gaps + 1)
        if len(groups) > 1 and (aa[0] + 360 - aa[-1]) < 6.0:        # wrap-around group
            groups[0] = np.r_[groups[-1], groups[0]]
            groups = groups[:-1]
        for g in groups:
            if len(g) < 8:
                continue
            c, r, rms = circle_fit(pts[g, :2])
            holes.append(dict(n=int(len(g)), centre_E_mm=np.round(c, 3).tolist(), diameter_mm=round(2 * r, 3),
                              rms_mm=round(rms, 4), bolt_circle_d_mm=round(2 * float(np.hypot(*c)), 3),
                              angle_E_deg=round(float(np.degrees(math.atan2(c[1], c[0]))), 2),
                              depth_mm=round(float(top - pts[g, 2].min()), 2)))
    out["holes_near_face"] = holes
    # outside the 63 mm flange body near the face: the tool connector
    out_sel = rho > 32.5
    if out_sel.any():
        Q = E[out_sel]
        c = Q.mean(0)
        out["outside_r32p5"] = dict(n=int(out_sel.sum()), centroid_E_mm=np.round(c, 2).tolist(),
                                    angle_E_deg=round(float(np.degrees(math.atan2(c[1], c[0]))), 2),
                                    angle_range_deg=[round(float(ang[out_sel].min()), 1), round(float(ang[out_sel].max()), 1)],
                                    z_range_E_mm=[round(float(Q[:, 2].min()), 2), round(float(Q[:, 2].max()), 2)],
                                    rho_range_mm=[round(float(rho[out_sel].min()), 2), round(float(rho[out_sel].max()), 2)])
    # radius census of the face ring (to see a 31.5 recess, a 63 mm rim)
    ring = np.abs(E[:, 2] - top) < 0.05
    rr, rc = np.unique(np.round(rho[ring], 1), return_counts=True)
    out["face_radii_mm"] = [(float(x), int(n)) for x, n in zip(rr, rc) if n >= 8]
    (HERE / "mesh_wrist3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    near = E[:, 2] > top - 25
    axs[0].scatter(E[near, 0], E[near, 1], s=2, c=E[near, 2], cmap="viridis")
    for h in holes:
        axs[0].add_patch(plt.Circle(h["centre_E_mm"], h["diameter_mm"] / 2, fill=False, color="r"))
        axs[0].annotate(f'{h["angle_E_deg"]:.0f} deg, d {h["diameter_mm"]:.1f}', h["centre_E_mm"], fontsize=9)
    if out_sel.any():
        axs[0].scatter(E[out_sel, 0], E[out_sel, 1], s=6, c="k", label="vertices outside r 32.5 mm")
    axs[0].arrow(0, 0, 40, 0, color="r", width=0.4)
    axs[0].arrow(0, 0, 0, 40, color="g", width=0.4)
    axs[0].text(42, 0, "x_E", color="r")
    axs[0].text(0, 42, "y_E", color="g")
    axs[0].set_aspect("equal")
    axs[0].set_title("Wrist-3 visual mesh in tool0, seen from +z_E (tool side), z > face - 25 mm")
    axs[0].legend(loc="lower left", fontsize=8)
    axs[1].scatter(E[:, 1], E[:, 2], s=2, c="gray")
    if out_sel.any():
        axs[1].scatter(E[out_sel, 1], E[out_sel, 2], s=6, c="k")
    axs[1].axhline(top, color="b", lw=0.8)
    axs[1].set_xlabel("y_E (mm)")
    axs[1].set_ylabel("z_E (mm)")
    axs[1].set_aspect("equal")
    axs[1].set_title("Side view along x_E")
    fig.tight_layout()
    fig.savefig(HERE / "mesh_wrist3.png", dpi=110)


if __name__ == "__main__":
    main()
