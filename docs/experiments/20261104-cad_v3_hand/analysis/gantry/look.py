#!/usr/bin/env python3
"""Scratch: three orthographic views of chosen parts of file "hand", coloured per part, labelled by part name.

    .venv/bin/python logs/20261104-cad_v3/gantry/look.py OUT.png name,name,...  [--xlim a b] [--ylim a b] [--zlim a b]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection

REPO = Path(__file__).resolve().parents[3]
PARTS = {p["name"]: p for p in json.loads((REPO / "logs/20261104-cad_v3/extract/parts.json").read_text())["parts"]
         if p["file"] == "hand"}

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("names")
ap.add_argument("--xlim", nargs=2, type=float)
ap.add_argument("--ylim", nargs=2, type=float)
ap.add_argument("--zlim", nargs=2, type=float)
ap.add_argument("--title", default="")
a = ap.parse_args()
names = a.names.split(",")
cmap = plt.get_cmap("tab20")
fig, axs = plt.subplots(1, 3, figsize=(24, 9))
views = [(0, 1, "top view (x_A right, y_A up)"), (0, 2, "front view (x_A right, z_A up)"),
         (1, 2, "side view (y_A right, z_A up)")]
for k, nm in enumerate(names):
    p = PARTS[nm]
    m = trimesh.load(REPO / p["stl"], force="mesh")
    c = cmap(k % 20)
    for ax, (i, j, t) in zip(axs, views):
        tri = m.vertices[m.faces][:, :, [i, j]]
        ax.add_collection(PolyCollection(tri, facecolors=[c], edgecolors="none", alpha=0.25))
        cc = np.asarray(p["com"])
        ax.text(cc[i], cc[j], nm, fontsize=9, ha="center", va="center", color="k")
for ax, (i, j, t) in zip(axs, views):
    ax.set_title(t, fontsize=14)
    ax.set_aspect("equal")
    ax.autoscale()
    lims = [a.xlim, a.ylim, a.zlim]
    if lims[i]:
        ax.set_xlim(*lims[i])
    if lims[j]:
        ax.set_ylim(*lims[j])
    ax.grid(True, lw=0.3)
    ax.tick_params(labelsize=12)
fig.suptitle(a.title, fontsize=15)
fig.tight_layout()
fig.savefig(a.out, dpi=80)
print("wrote", a.out)
