#!/usr/bin/env python3
"""Four and five magnets in concept b's 11/22 pad (2026-10-20 evening): the layouts over the board's five chips (palmar
view) and the readout model's force resolution and magnet travel against the magnets' depth under the apex, from
docs/experiments/20261019-v2_tip_fe_checks/sensing.json (scripts/v2_tip_fe_checks.py --sense b_v2*).

Owner, 2026-10-20 18:10: "with the 5 magnets, we have to lower the magnets farther away from the tip / closer to the
magnetometer. it looks like it wont deform very much. can you also re-prepare the 4 magnets configuration? did you measure
this difference much?" Output: docs/experiments/20261018-dip_link_board_mount/v3/b/media/20261020-dip_link_v3b_magnet_layouts.png

    .venv/bin/python scripts/dip_b_magnet_layouts_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SENS = ROOT / "docs/experiments/20261019-v2_tip_fe_checks/sensing.json"
OUT = ROOT / "docs/experiments/20261018-dip_link_board_mount/v3/b/media/20261020-dip_link_v3b_magnet_layouts.png"
CHIPS = {"U1": (0.0, 0.0), "U2": (0.0, 7.0), "U3": (7.0, 0.0), "U4": (0.0, -7.0), "U5": (-7.0, 0.05)}   # mlx90393_model
FIVE = [(sy * 2.5, -0.5 + sz * 4.9) for sy in (-1, 1) for sz in (-1, 1)] + [(0.0, -0.5)]   # y, z from U1 (contact height)
FOUR = [(sy * 3.0, sz * 3.5) for sy in (-1, 1) for sz in (-1, 1)]
CASES = {"five": [("bm_five_d65", 6.5), ("bm_five_d70", 7.0), ("bm_five_d75", 7.5), ("bm_five_d85", 8.5)],
         "four": [("bm_four_d70", 7.0), ("bm_four_d75", 7.5)]}          # grid bmag (scripts/v2_tip_fe_checks.py)
ONE_WALL = {"five": "bm_five_d75_w1", "four": "bm_four_d75_w1"}         # one perimeter along each pocket, 7.5 mm
COL = {"five": "#1f4e9c", "four": "#b5442a"}


def outline():
    """The pad's widest section (the straight flank) in the palmar view: half-width 7.4 mm, 22 mm toward the mount face
    and 11 mm toward the tip from the contact height; the mount face 10.86 mm and the tip end 10.55 mm from it."""
    z = np.linspace(-10.86, 10.55, 400)
    a = np.where(z < 0, 22.0, 11.0)
    y = 7.4 * np.sqrt(np.clip(1 - (z / a) ** 2, 0, None))
    return np.r_[y, -y[::-1], y[:1]], np.r_[z, z[::-1], z[:1]]


def main():
    rows = {r["case"]: r for r in json.loads(SENS.read_text())["rows"]}
    plt.rcParams.update({"font.family": "serif", "font.size": 14})
    fig = plt.figure(figsize=(11.0, 10.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.32, wspace=0.30)
    oy, oz = outline()
    for k, (lay, mags, title) in enumerate((("five", FIVE, "Five magnets (B00)"), ("four", FOUR, "Four magnets (B4)"))):
        ax = fig.add_subplot(gs[0, k])
        ax.plot(oy, oz, color="0.3", lw=1.2)
        for n, (u, v) in CHIPS.items():
            ax.add_patch(plt.Rectangle((u - 1.5, v - 1.5), 3.0, 3.0, fc="0.85", ec="0.35", lw=0.8, zorder=1))
            dx, dy, ha, va = (-1.6, 1.6, "right", "bottom") if n == "U2" else ((-1.6, -1.6, "right", "top") if n == "U5"
                                                                                else (1.6, -1.6, "left", "top"))
            ax.text(u + dx, v + dy, n, ha=ha, va=va, fontsize=13, color="0.2", zorder=3)
        for y, z in mags:
            ax.add_patch(plt.Circle((y, z), 1.5, fc=COL[lay], ec="k", lw=0.6, alpha=0.75, zorder=2))
        ax.set_aspect("equal")
        ax.set_xlim(-10.5, 10.5)
        ax.set_ylim(-12.5, 12.5)
        ax.set_title(title, fontsize=15)
        ax.set_xlabel("Across the finger, y (mm)")
        ax.set_ylabel("Along the finger from U1 (mm)")
    axr = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[1, 1])
    for lay, cs in CASES.items():
        have = [(d, rows[c]) for c, d in cs if c in rows]
        if not have:
            continue
        lbl = {"five": "Five magnets", "four": "Four magnets"}[lay]
        ok = [h for h in have if h[1]["gain_sel"] is not None]          # a gain setting reads the field
        axr.plot([h[0] for h in ok], [1e3 * h[1]["gain"]["centre_force_N"] for h in ok], "o-", color=COL[lay], label=lbl,
                 lw=1.6, ms=7)
        for dd, r in have:
            if r["gain_sel"] is None:
                axr.text(8.75, 36.0, f"{dd:.1f} mm: {r['B_max_component_mT']:.0f} mT at a chip,\nno gain setting reads it",
                         ha="right", va="bottom", fontsize=12, color=COL[lay])
        axt.plot([h[0] for h in have], [h[1]["magnet_travel_um_per_N"] for h in have], "o-", color=COL[lay], label=lbl,
                 lw=1.6, ms=7)
        w = rows.get(ONE_WALL[lay])
        if w is not None:                                               # open marker: one wall around the pockets
            axr.plot([7.5], [1e3 * w["gain"]["centre_force_N"]], "o", mfc="white", mec=COL[lay], mew=1.8, ms=9,
                     label=f"{lbl}, one wall")
            axt.plot([7.5], [w["magnet_travel_um_per_N"]], "o", mfc="white", mec=COL[lay], mew=1.8, ms=9,
                     label=f"{lbl}, one wall")
    axr.set_ylabel("Force resolution at the contact point (mN)")
    axr.set_title("Readout model", fontsize=15)
    axt.set_ylabel("Magnet travel at 1 N (µm)")
    axt.set_title("Finite elements", fontsize=15)
    for ax in (axr, axt):
        ax.set_xlabel("Magnet depth under the apex (mm)")
        ax.grid(alpha=0.3)
        ax.axvline(7.5, color="0.5", ls=":", lw=1.0)
        ax.set_xlim(6.2, 8.8)
    axr.set_ylim(0, 60)
    axt.set_ylim(0, 9)
    axr.legend(loc="lower left", fontsize=12, frameon=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight", facecolor="white")
    print("->", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
