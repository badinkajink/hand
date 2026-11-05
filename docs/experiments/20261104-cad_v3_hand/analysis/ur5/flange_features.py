#!/usr/bin/env python3
"""The tool flange of the UR5 (CB3) wrist 3 visual mesh in tool0 axes: the flange face, its bores and holes (circles
fitted to the side walls of each pocket), the tool connector, and a picture of the face seen from the tool side.

tool0 coincides with wrist_3_link in ur_description's ur5.urdf (flange rpy (0, -pi/2, -pi/2) then tool0 rpy (pi/2, 0,
pi/2) compose to the identity), so the visual mesh goes into tool0 by its URDF visual origin alone.

    ~/.claude/bin/resguard.sh run --mem 3G -- .venv/bin/python logs/20261104-cad_v3/ur5/flange_features.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import trimesh

ROOT = Path("/home/humanoid/Programs/hand")
sys.path.insert(0, str(ROOT / "src"))
from sr2hand.geometry import rotations as R  # noqa: E402

HERE = Path(__file__).resolve().parent
UR = Path.home() / "Programs/rh56_controller/magpie_force_control/src/ur_description"
T_VIS = R.tf(R.euler_to_mat([np.pi / 2, 0.0, 0.0], "xyz"), np.array([0.0, 0.0, -0.0818]))   # wrist_3_link visual origin


def fit_circle(P):
    """Least-squares circle (cx, cy, r) through 2-D points (Kasa)."""
    A = np.c_[2 * P[:, 0], 2 * P[:, 1], np.ones(len(P))]
    b = (P ** 2).sum(1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return cx, cy, float(np.sqrt(c + cx ** 2 + cy ** 2))


def main():
    parts = json.loads((HERE / "meshes/visual/materials.json").read_text())["wrist3"]["parts"]
    meshes = []
    for p in parts:
        m = trimesh.load(HERE / "meshes/visual" / p["file"], force="mesh", process=False)
        m.apply_transform(T_VIS)
        m.apply_scale(1e3)                                     # mm, tool0 axes
        meshes.append((p, m))
    allm = trimesh.util.concatenate([m for _, m in meshes])
    allm.merge_vertices()
    z_face = float(allm.vertices[:, 2].max())
    out = dict(frame="tool0 = wrist_3_link (mm); z out of the flange", z_face_mm=round(z_face, 3),
               bounds_mm=np.round(allm.bounds, 2).tolist())
    # faces in the flange plane and their radial extent
    n = allm.face_normals
    c = allm.triangles_center
    top = (n[:, 2] > 0.99) & (np.abs(c[:, 2] - z_face) < 0.05)
    V = allm.triangles[top].reshape(-1, 3)
    rr = np.hypot(V[:, 0], V[:, 1])
    out["face_annulus_mm"] = [round(float(rr.min()), 2), round(float(rr.max()), 2)]
    # side walls of pockets that open on the face: faces with horizontal normals whose top touches the face plane
    side = (np.abs(n[:, 2]) < 0.2) & (allm.triangles[:, :, 2].max(1) > z_face - 0.05) & \
           (np.hypot(c[:, 0], c[:, 1]) < 36.0)
    sub = allm.submesh([np.nonzero(side)[0]], append=True)
    comps = sub.split(only_watertight=False)
    holes = []
    for comp in comps:
        P = comp.vertices
        if len(P) < 6:
            continue
        cx, cy, r = fit_circle(P[:, :2])
        res = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r)
        # concave (a hole) when the wall normals point toward the circle's centre
        cc = comp.triangles_center
        radial = np.c_[cc[:, 0] - cx, cc[:, 1] - cy]
        inward = float(np.mean(np.einsum("ij,ij->i", comp.face_normals[:, :2], radial) < 0))
        holes.append(dict(centre_mm=[round(cx, 2), round(cy, 2)], r_mm=round(r, 2), d_mm=round(2 * r, 2),
                          depth_mm=round(float(z_face - P[:, 2].min()), 2),
                          pcd_r_mm=round(float(np.hypot(cx, cy)), 2),
                          angle_deg_from_x=round(float(np.degrees(np.arctan2(cy, cx))), 1),
                          fit_rms_mm=round(float(np.sqrt((res ** 2).mean())), 3), concave_frac=round(inward, 2),
                          n_vertices=int(len(P))))
    holes.sort(key=lambda h: (h["pcd_r_mm"], h["angle_deg_from_x"]))
    out["pockets_on_face"] = holes
    # the radial protrusion beyond the 37.5 mm housing (the tool connector)
    Vall = allm.vertices
    rad = np.hypot(Vall[:, 0], Vall[:, 1])
    bump = Vall[rad > 37.6]
    if len(bump):
        cxy = bump[:, :2].mean(0)
        out["protrusion"] = dict(n_vertices=int(len(bump)), mean_xy_mm=np.round(cxy, 2).tolist(),
                                 angle_deg_from_x=round(float(np.degrees(np.arctan2(cxy[1], cxy[0]))), 1),
                                 r_max_mm=round(float(rad.max()), 2),
                                 z_range_mm=[round(float(bump[:, 2].min()), 2), round(float(bump[:, 2].max()), 2)])
    print(json.dumps(out, indent=1))
    (HERE / "flange_features.json").write_text(json.dumps(out, indent=1))

    # picture: the face seen from the tool side (+z toward the viewer), x right, y up; shaded by the normal's z
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.6))
    for ax, (title, sel) in zip(axs, (("Flange face of wrist 3 seen from the tool, tool0 axes (mm)", None),
                                      ("Side view along tool0 x: the connector on -y", "side"))):
        tri = allm.triangles
        nz = allm.face_normals
        if sel is None:
            order = np.argsort(tri[:, :, 2].mean(1))           # painter's order, far faces first
            polys = tri[order][:, :, :2]
            shade = np.clip(0.25 + 0.75 * np.abs(nz[order, 2]), 0, 1)
            depth = (tri[order][:, :, 2].mean(1) - z_face)
            col = plt.cm.Greys_r(0.15 + 0.7 * shade * np.clip(1 + depth / 40.0, 0.2, 1))
        else:
            order = np.argsort(tri[:, :, 0].mean(1))
            polys = tri[order][:, :, [1, 2]]
            shade = np.clip(0.25 + 0.75 * np.abs(nz[order, 0]), 0, 1)
            col = plt.cm.Greys_r(0.15 + 0.7 * shade)
        ax.add_collection(PolyCollection(polys, facecolors=col, edgecolors="none"))
        ax.set_aspect("equal")
        ax.grid(True, lw=0.3)
        if sel is None:
            ax.set_xlim(-46, 46)
            ax.set_ylim(-48, 46)
            th = np.linspace(0, 2 * np.pi, 361)
            ax.plot(25 * np.cos(th), 25 * np.sin(th), "--", color="tab:orange", lw=1, label="50 mm bolt circle")
            ax.annotate("", xy=(30, 0), xytext=(0, 0), arrowprops=dict(color="red", width=1.5, headwidth=8))
            ax.annotate("", xy=(0, 30), xytext=(0, 0), arrowprops=dict(color="green", width=1.5, headwidth=8))
            ax.text(31, -3.5, "x", color="red", fontsize=14)
            ax.text(1.5, 31, "y", color="green", fontsize=14)
            for h in holes:
                if h["pcd_r_mm"] > 5:
                    ax.plot(*h["centre_mm"], "+", color="tab:blue", ms=10)
                    ang = f"{h['angle_deg_from_x']:.0f}".replace("-", "\u2212")
                    ax.text(h["centre_mm"][0] + 2, h["centre_mm"][1] + 2,
                            f"\u00f8{h['d_mm']:.1f} mm\n{ang} deg", color="tab:blue", fontsize=9)
            # the ISO 9409-1 dowel hole is not in the mesh; UR's UR5e drawing puts it on the connector's side
            ax.add_patch(plt.Circle((0.0, -25.0), 3.0, fill=False, ls=":", color="tab:purple", lw=1.5))
            ax.text(4.0, -27.5, "\u00f86 dowel hole expected here\n(not in the mesh)", color="tab:purple", fontsize=9)
            ax.text(5.0, -44.0, "tool connector", color="black", fontsize=9)
            ax.set_xlabel("x of tool0 (mm)")
            ax.set_ylabel("y of tool0 (mm)")
            ax.legend(loc="lower right", fontsize=9)
        else:
            ax.set_xlim(-48, 46)
            ax.set_ylim(-40, 6)
            ax.set_xlabel("y of tool0 (mm)")
            ax.set_ylabel("z of tool0 (mm), the face at 0")
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "flange_tool0.png", dpi=90)


if __name__ == "__main__":
    main()
