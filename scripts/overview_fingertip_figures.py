#!/usr/bin/env python3
"""Figures of the printed-fingertips overview that no result page draws as an image (scripts/overview_fingertip_text.py).

    .venv/bin/python scripts/overview_fingertip_figures.py

moduli: the homogenized moduli of the printed gyroid against solid fraction, from the cells of the finite-element study
(docs/experiments/20261009-tpu_tip_fem/cells.jsonl): the slicer's gyroid in the layer plane (Ex) and along the build
axis (Ez), the ideal sheet gyroid, eFlesh's cut cells, the power-law fits and the Gibson-Ashby limits; the infill
settings of the printed pads (30 and 50 %) marked. Written to docs/overviews/media/fingertip_moduli_plot.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "docs/experiments/20261009-tpu_tip_fem/cells.jsonl"
OUT = ROOT / "docs/overviews/media/fingertip_moduli_plot.png"


def moduli(out=OUT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 14, "axes.labelsize": 14, "legend.fontsize": 12.5,
                         "xtick.labelsize": 13, "ytick.labelsize": 13})
    rows = [json.loads(l) for l in open(CELLS)]
    sl = sorted((r for r in rows if r["kind"] == "slicer" and abs(r["h_mm"] - 0.05) < 1e-9), key=lambda r: r["rho"])
    sl10 = sorted((r for r in rows if r["kind"] == "slicer" and abs(r["h_mm"] - 0.1) < 1e-9), key=lambda r: r["rho"])
    sh = sorted((r for r in rows if r["kind"] == "sheet" and abs(r["h_mm"] - 0.1) < 1e-9), key=lambda r: r["rho"])
    ef = [r for r in rows if r["kind"] == "eflesh"]
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=100)
    phi = np.linspace(0.08, 0.6, 200)
    ax.plot(phi, phi ** 2, color="#999", ls="--", lw=1.2, label="Gibson-Ashby, bending: $\\phi^2$")
    ax.plot(phi, phi / 3, color="#999", ls=":", lw=1.4, label="Gibson-Ashby, stretching: $\\phi/3$")
    ax.plot(phi, 0.43 * phi ** 1.40, color="#4A7FC4", lw=1.0, alpha=0.6)
    ax.plot(phi, 0.60 * phi ** 1.29, color="#BE7514", lw=1.0, alpha=0.6)
    for data, mk, lab_x, lab_z in ((sl10, "o", "Slicer gyroid, layer plane $E_x$", "Slicer gyroid, build axis $E_z$"),
                                    (sl, "s", None, None)):
        x = [r["rho"] for r in data]
        ax.plot(x, [r["Ex"] for r in data], mk, color="#4A7FC4", ms=7, mfc="#4A7FC4" if mk == "o" else "none",
                label=lab_x)
        ax.plot(x, [r["Ez"] for r in data], mk, color="#BE7514", ms=7, mfc="#BE7514" if mk == "o" else "none",
                label=lab_z)
    ax.plot([r["rho"] for r in sh], [r["Ex"] for r in sh], "^", color="#5E8F5A", ms=7, label="Ideal sheet gyroid (isotropic)")
    ax.plot([r["rho"] for r in ef], [r["Ex"] for r in ef], "D", color="#000", ms=6, mfc="none", label="eFlesh cut cells")
    for r in sl:
        if round(r["rho_req"], 2) in (0.3, 0.5):
            ax.annotate(f"setting {r['rho_req']:.0%}: {r['rho']:.1%} solid", (r["rho"], r["Ez"]),
                        xytext=(8, 10) if r["rho_req"] < 0.4 else (-190, 12), textcoords="offset points", fontsize=12.5)
    ax.set_yscale("log")
    ax.set_xlabel("Solid fraction $\\phi$ of the core")
    ax.set_ylabel("Modulus / solid TPU modulus $E_s$")
    ax.set_xlim(0.08, 0.6)
    ax.set_ylim(0.004, 0.5)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    moduli()
