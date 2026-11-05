#!/usr/bin/env python3
"""Each UR5 link's collision mesh in its URDF link frame (mm), three orthographic views, to fit collision primitives.

    ~/.claude/bin/resguard.sh run --mem 3G -- .venv/bin/python logs/20261104-cad_v3/ur5/link_views.py
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

ROOT = Path("/home/humanoid/Programs/hand")
sys.path.insert(0, str(ROOT / "src"))
from morphohand.geometry import rotations as R  # noqa: E402

HERE = Path(__file__).resolve().parent
UR = Path.home() / "Programs/rh56_controller/magpie_force_control/src/ur_description"
LINKS = ("base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link")


def origin(el):
    o = el.find("origin")
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0.0] * 3
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0.0] * 3
    return R.tf(R.euler_to_mat(rpy, "xyz"), np.array(xyz))


def link_meshes():
    root = ET.parse(UR / "ur5.urdf").getroot()
    out = {}
    for ln in root.findall("link"):
        col = ln.find("collision")
        if col is None:
            continue
        m = trimesh.load(UR / col.find("geometry/mesh").get("filename").replace("ur_description/", ""), force="mesh")
        m.apply_transform(origin(col))
        m.apply_scale(1e3)
        out[ln.get("name")] = m
    return out


def main():
    ms = link_meshes()
    fig, axs = plt.subplots(len(LINKS), 3, figsize=(13, 4.0 * len(LINKS)))
    for i, ln in enumerate(LINKS):
        m = ms[ln]
        P = trimesh.sample.sample_surface(m, 20000, seed=0)[0]
        print(ln, "bounds mm", np.round(m.bounds, 1).tolist())
        for j, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            ax = axs[i, j]
            ax.scatter(P[:, a], P[:, b], s=0.2, c="k", alpha=0.3)
            ax.axhline(0, color="r", lw=0.5)
            ax.axvline(0, color="r", lw=0.5)
            ax.set_aspect("equal")
            ax.set_xlabel("xyz"[a] + " (mm)")
            ax.set_ylabel("xyz"[b] + " (mm)")
            ax.set_title(f"{ln}: {'xyz'[a]}{'xyz'[b]} view", fontsize=10)
            ax.grid(True, lw=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "link_views.png", dpi=70)


if __name__ == "__main__":
    main()
