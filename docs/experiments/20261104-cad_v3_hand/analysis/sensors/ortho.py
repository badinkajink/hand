#!/usr/bin/env python3
"""Scratch: orthographic views of chosen solids of the hand file, in frame A (mm), painter's algorithm with flat shading.
Usage: ortho.py OUT.png 'name:color,...' view[,view...] [xlo,xhi,ylo,yhi,zlo,zhi crop] [context names]
views: top (look -z; screen x_A, y_A), bottom (look +z; screen x_A, -y_A), front (look +y; x_A, z_A),
       back (look -y; -x_A, z_A), right (look -x; y_A, z_A), left (look +x; -y_A, z_A)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb

REPO = Path("/home/humanoid/Programs/hand")
EX = REPO / "logs/20261104-cad_v3/extract"
VIEWS = {  # (screen-x axis, screen-y axis, viewing direction)
    "top": ((1, 0, 0), (0, 1, 0), (0, 0, -1)),
    "bottom": ((1, 0, 0), (0, -1, 0), (0, 0, 1)),
    "front": ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "back": ((-1, 0, 0), (0, 0, 1), (0, -1, 0)),
    "right": ((0, 1, 0), (0, 0, 1), (-1, 0, 0)),
    "left": ((0, -1, 0), (0, 0, 1), (1, 0, 0)),
}
LABEL = {(1, 0, 0): "x_A", (-1, 0, 0): "-x_A", (0, 1, 0): "y_A", (0, -1, 0): "-y_A", (0, 0, 1): "z_A"}


def load(names, file="hand"):
    out = {}
    for n in names:
        out[n] = trimesh.load(EX / file / f"{n}.stl", process=False)
    return out


def draw(ax, meshes, colors, view, crop=None, light=(0.3, 0.5, 0.8), alpha=1.0):
    sx, sy, vd = (np.array(v, float) for v in VIEWS[view])
    polys, cols, depth = [], [], []
    L = np.array(light, float)
    L /= np.linalg.norm(L)
    for n, m in meshes.items():
        tri = m.vertices[m.faces]
        if crop is not None:
            c = tri.mean(axis=1)
            keep = np.all((c >= crop[0::2]) & (c <= crop[1::2]), axis=1)
            tri = tri[keep]
        if len(tri) == 0:
            continue
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nn = np.linalg.norm(nrm, axis=1)
        ok = nn > 1e-12
        tri, nrm = tri[ok], nrm[ok] / nn[ok, None]
        facing = nrm @ (-vd) > 0
        tri, nrm = tri[facing], nrm[facing]
        shade = 0.45 + 0.55 * np.clip(np.abs(nrm @ L), 0, 1)
        base = np.array(to_rgb(colors.get(n, "0.8")))
        polys.append(np.stack([tri @ sx, tri @ sy], axis=-1))
        cols.append(np.clip(base[None, :] * shade[:, None], 0, 1))
        depth.append(tri.mean(axis=1) @ vd)
    if not polys:
        return
    P = np.concatenate(polys)
    C = np.concatenate(cols)
    D = np.concatenate(depth)
    order = np.argsort(-D)                       # far first
    ax.add_collection(PolyCollection(P[order], facecolors=C[order], edgecolors=C[order], linewidths=0.2,
                                     alpha=alpha))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel(LABEL[tuple(int(v) for v in sx)] + " (mm)", fontsize=13)
    ax.set_ylabel(LABEL[tuple(int(v) for v in sy)] + " (mm)", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.grid(True, lw=0.3)
    ax.set_title(f"{view} view (looking along {LABEL.get(tuple(int(v) for v in vd), str(vd))})", fontsize=13)


if __name__ == "__main__":
    outp = sys.argv[1]
    colors = {}
    for tok in sys.argv[2].split(","):
        n, c = tok.split(":", 1)
        colors[n] = c
    views = sys.argv[3].split(",")
    crop = np.array([float(x) for x in sys.argv[4].split(",")]) if len(sys.argv) > 4 and sys.argv[4] else None
    ctx = sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[5] else []
    names = list(colors) + [c for c in ctx if c not in colors]
    meshes = load(names)
    cols = min(2, len(views))
    rows = (len(views) + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows))
    axs = np.atleast_1d(axs).ravel()
    for a, v in zip(axs, views):
        draw(a, meshes, colors, v, crop)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, list(colors), loc="lower center", ncol=min(8, len(colors)), fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(outp, dpi=80)
    print("wrote", outp)
