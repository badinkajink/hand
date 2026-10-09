#!/usr/bin/env python3
"""Still images for the TPU tip page: the printed insert's wall, skin and core classes for both print orientations, and
slices of the resolved 20 % gyroid tip (femtip env; reads the caches of tpu_tip_fem_tip.py).

    ~/miniconda3/envs/femtip/bin/python scripts/tpu_tip_fem_media.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_tip_fem_tip as T  # noqa: E402

OUT = ROOT / "docs/experiments/20261009-tpu_tip_fem/media"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 13})
    cmap = ListedColormap(["white", "#2a78d6", "#eb6834", "#d9d9d9"])
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 9.0))
    for r, (b, lab) in enumerate((("x", "Layers across the load"), ("z", "Layers along the load"))):
        cls, o, h = T.classify(b, cache=T.AUX_CACHE / f"classes_build{b}_h0.05.npz")
        xs = o[0] + (np.arange(cls.shape[0]) + .5) * h
        ys = o[1] + (np.arange(cls.shape[1]) + .5) * h
        zs = o[2] + (np.arange(cls.shape[2]) + .5) * h
        k = int(np.argmin(abs(zs - T.Z_C)))
        j = int(np.argmin(abs(ys)))
        axs[r, 0].imshow(cls[:, :, k].T, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap=cmap, vmin=0, vmax=3,
                         interpolation="nearest")
        axs[r, 0].set_title(f"{lab}: section at the contact")
        axs[r, 0].set_xlabel("Depth x (mm), palmar face right")
        axs[r, 0].set_ylabel("Across y (mm)")
        axs[r, 1].imshow(cls[:, j, :].T, origin="lower", extent=[xs[0], xs[-1], zs[0], zs[-1]], cmap=cmap, vmin=0, vmax=3,
                         interpolation="nearest")
        axs[r, 1].set_title(f"{lab}: section y = 0")
        axs[r, 1].set_xlabel("Depth x (mm), palmar face right")
        axs[r, 1].set_ylabel("Along the finger z (mm)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#2a78d6", "#eb6834", "#d9d9d9")]
    fig.legend(handles, ["Perimeter walls", "Top and bottom skins", "Infill core"], loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "20261009-classes.png", dpi=80)
    plt.close(fig)
    D = np.load(ROOT / "assets/fem/20261009-tpu_tip/voxels/g20x_s0.npz")
    occ, o, h = D["occ"], D["origin"], float(D["h"])
    xs = o[0] + (np.arange(occ.shape[0]) + .5) * h
    ys = o[1] + (np.arange(occ.shape[1]) + .5) * h
    zs = o[2] + (np.arange(occ.shape[2]) + .5) * h
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.8))
    i = int(np.argmin(abs(xs - 6.0)))
    axs[0].imshow(occ[i].T, origin="lower", extent=[ys[0], ys[-1], zs[0], zs[-1]], cmap="gray_r", interpolation="nearest")
    axs[0].set_title("Layer at x = 6 mm")
    axs[0].set_xlabel("Across y (mm)")
    axs[0].set_ylabel("Along the finger z (mm)")
    k = int(np.argmin(abs(zs - T.Z_C)))
    axs[1].imshow(occ[:, :, k].T, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="gray_r", interpolation="nearest")
    axs[1].set_title("Section at the contact")
    axs[1].set_xlabel("Depth x (mm)")
    axs[1].set_ylabel("Across y (mm)")
    j = int(np.argmin(abs(ys)))
    axs[2].imshow(occ[:, j, :].T, origin="lower", extent=[xs[0], xs[-1], zs[0], zs[-1]], cmap="gray_r", interpolation="nearest")
    axs[2].set_title("Section y = 0")
    axs[2].set_xlabel("Depth x (mm)")
    axs[2].set_ylabel("Along the finger z (mm)")
    fig.tight_layout()
    fig.savefig(OUT / "20261009-voxels_g20x.png", dpi=80)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
