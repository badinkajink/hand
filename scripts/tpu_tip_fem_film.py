#!/usr/bin/env python3
"""Film of the resolved printed tip under the screwdriver cylinder: the section through the contact (y = 0) with the
voxels displaced (magnified) and coloured by displacement, beside the pressure on the palmar face, at approaches from
0 to the given maximum (GPU solver of tpu_tip_fem_gpu.py, warm-started from frame to frame).

    .venv/bin/python scripts/tpu_tip_fem_film.py VOXELS.npz --dmax 0.13 --frames 13 --out media/film.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_tip_fem_gpu as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("voxels")
    ap.add_argument("--dmax", type=float, default=0.13)
    ap.add_argument("--frames", type=int, default=13)
    ap.add_argument("--E", type=float, default=30.0)
    ap.add_argument("--mag", type=float, default=15.0)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data")
    ap.add_argument("--surface", action="store_true", help="contact gaps from the STL surface instead of the voxel tops")
    ap.add_argument("--from-data", action="store_true", help="render the frames saved in --data without solving again")
    a = ap.parse_args()
    D = np.load(a.voxels, allow_pickle=True)
    occ, origin, h = D["occ"], D["origin"], float(D["h"])
    if a.from_data:
        S = np.load(a.data)
        frames = [dict(delta=float(d), F=float(f), p=p, u_sec=u) for d, f, p, u in zip(S["deltas"], S["F"], S["p"], S["u_sec"])]
        render(a, frames, S["X_corners"], S["X_face"], origin, occ.shape, h)
        return
    tip = G.GpuTip(occ, origin, h, a.E)
    nodes = G.face_nodes(tip.X, tip.fixed_dof[0::3])
    Xg = tip.X[nodes].copy()
    if a.surface:
        Xg[:, 0], _ = G.true_surface_x(Xg)
    g0 = G.gap0(Xg, "cyl")
    # section voxels: the layer of voxels whose lower y face is at y = 0
    j0 = int(round((0.0 - origin[1]) / h))
    sec = np.argwhere(occ[:, j0, :])                       # (i, k)
    ny1, nz1 = occ.shape[1] + 1, occ.shape[2] + 1
    # compact node ids of the 4 corners (y = j0 plane) of each section voxel
    used_ids = {}
    gi = np.round((tip.X[:, 0] - origin[0]) / h).astype(int)
    gj = np.round((tip.X[:, 1] - origin[1]) / h).astype(int)
    gk = np.round((tip.X[:, 2] - origin[2]) / h).astype(int)
    key = (gi * ny1 + gj) * nz1 + gk
    lut = dict(zip(key.tolist(), range(len(key))))
    corners = np.array([[lut[((i + di) * ny1 + j0) * nz1 + (k + dk)] for di, dk in ((0, 0), (1, 0), (1, 1), (0, 1))]
                        for i, k in sec])
    deltas = np.linspace(0, a.dmax, a.frames + 1)[1:]
    frames = []
    x, active = None, None
    for dl in deltas:
        p, x, active, info = G.contact(tip, nodes, g0, dl, x0=x, active0=active)
        u = x.view(-1, 3).cpu().numpy()
        frames.append(dict(delta=dl, F=float(p.sum()), p=p.copy(), u_sec=u[corners].astype(np.float32)))
        print(f"  delta {dl * 1e3:.1f} um: F {p.sum():.3f} N", flush=True)
    if a.data:
        np.savez_compressed(a.data, deltas=deltas, F=np.array([f["F"] for f in frames]), X_face=tip.X[nodes],
                            p=np.array([f["p"] for f in frames]), sec=sec, X_corners=tip.X[corners],
                            u_sec=np.array([f["u_sec"] for f in frames]))
    render(a, frames, tip.X[corners], tip.X[nodes], origin, occ.shape, h)


def render(a, frames, X_corners, Xf, origin, shape, h):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    Xc = X_corners[:, :, [2, 0]]                            # (n, 4, [z, x])
    pmax = max(f["p"].max() for f in frames)
    umax = max(np.abs(f["u_sec"]).max() for f in frames)
    with tempfile.TemporaryDirectory() as td:
        for n, f in enumerate([dict(delta=0.0, F=0.0, p=np.zeros(len(Xf)), u_sec=np.zeros_like(frames[0]["u_sec"]))] + frames):
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 5.0), gridspec_kw=dict(width_ratios=[1.25, 1]))
            disp = Xc + a.mag * f["u_sec"][:, :, [2, 0]]
            mag = np.linalg.norm(f["u_sec"], axis=2).mean(1)
            pc = PolyCollection(disp, array=mag * 1e3, cmap="viridis", edgecolors="none", clim=(0, umax * 1e3))
            a1.add_collection(pc)
            zz = np.linspace(G.Z_C - 6, G.Z_C + 6, 200)
            xs = G.X_FACE - a.mag * f["delta"] + (G.R_TOOL - np.sqrt(G.R_TOOL ** 2 - (zz - G.Z_C) ** 2))
            a1.fill_between(zz, xs, xs + 4, color="0.8", alpha=0.3, lw=0)
            a1.plot(zz, xs, color="0.3", lw=1.2)
            a1.set_xlim(origin[2] - 0.5, origin[2] + shape[2] * h + 0.5)
            a1.set_ylim(origin[0] - 0.5, G.X_FACE + 4.5)
            a1.set_aspect("equal")
            a1.set_xlabel("Along the finger z (mm)", fontsize=15)
            a1.set_ylabel("Depth x (mm)", fontsize=15)
            a1.set_title(f"Section y = 0, displacement \u00d7{a.mag:g}", fontsize=15)
            cb = fig.colorbar(pc, ax=a1, shrink=0.8)
            cb.set_label("Displacement (\u00b5m)", fontsize=14)
            sc = a2.scatter(Xf[:, 2], Xf[:, 1], c=f["p"] / (h * h), s=3, cmap="magma", vmin=0, vmax=pmax / (h * h), marker="s")
            a2.set_xlim(G.Z_C - 4, G.Z_C + 4)
            a2.set_ylim(-7.6, 7.6)
            a2.set_aspect("equal")
            a2.set_xlabel("Along the finger z (mm)", fontsize=15)
            a2.set_ylabel("Across y (mm)", fontsize=15)
            a2.set_title("Pressure on the palmar face", fontsize=15)
            cb2 = fig.colorbar(sc, ax=a2, shrink=0.8)
            cb2.set_label("Pressure (MPa)", fontsize=14)
            fig.suptitle(f"{a.title}: approach {f['delta'] * 1e3:5.1f} \u00b5m, force {f['F']:.2f} N", fontsize=16)
            for ax in (a1, a2, cb.ax, cb2.ax):
                ax.tick_params(labelsize=13)
            fig.tight_layout()
            fig.savefig(f"{td}/f{n:03d}.png", dpi=100)
            plt.close(fig)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "2", "-i", f"{td}/f%03d.png", "-vf",
                        "scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=25", "-c:v", "libx264", "-crf", "27", "-pix_fmt", "yuv420p",
                        a.out], check=True)
        subprocess.run(["cp", f"{td}/f{len(frames):03d}.png", str(Path(a.out).with_suffix(".png"))], check=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
