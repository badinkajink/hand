#!/usr/bin/env python3
"""Scratch: six orthographic views of one STL (its own frame, mm)."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
sys.path.insert(0, "/home/humanoid/Programs/hand/logs/20261104-cad_v3/sensors")
from ortho import draw
m = trimesh.load(sys.argv[1], process=False)
views = sys.argv[3].split(",") if len(sys.argv) > 3 else ["top", "bottom", "front", "back", "right", "left"]
fig, axs = plt.subplots(2, 3, figsize=(21, 13))
for a, v in zip(axs.ravel(), views):
    draw(a, {"m": m}, {"m": "tab:orange"}, v)
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=70)
print("wrote", sys.argv[2])
