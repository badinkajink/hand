#!/usr/bin/env python3
"""Analyze full_range_sweep.py output: friction and gravity/bias vs joint angle.

    python3 analyze_full_range.py full_range_raw_<...>.txt [BIN_DEG]

For each joint, samples from the 'up' and 'down' legs (settle periods excluded) are binned by
measured position. Per bin:
  friction = (mean load_down - mean load_up) / 2   (load sign: negative pushes toward +)
  bias     = (mean load_up + mean load_down) / 2   (gravity, cable/spring, controller offset)
Positive friction = the servo had to push against the motion in both directions.
Writes <stem>_bins.csv and <stem>.png.
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze_friction_sweep import parse  # noqa: E402


def bins_for(rows: list[dict], width: float) -> list[dict]:
    up = [x for x in rows if x["phase"] == "up"]
    dn = [x for x in rows if x["phase"] == "down"]
    if not up or not dn:
        return []
    lo = max(min(x["pos"] for x in up), min(x["pos"] for x in dn))
    hi = min(max(x["pos"] for x in up), max(x["pos"] for x in dn))
    out = []
    a = lo
    while a + width <= hi + 1e-9:
        b = a + width
        lu = [x["load"] for x in up if a <= x["pos"] < b]
        ld = [x["load"] for x in dn if a <= x["pos"] < b]
        eu = [x["target"] - x["pos"] for x in up if a <= x["pos"] < b]
        ed = [x["pos"] - x["target"] for x in dn if a <= x["pos"] < b]
        if len(lu) >= 5 and len(ld) >= 5:
            mu, md = st.fmean(lu), st.fmean(ld)
            out.append(dict(center=round((a + b) / 2, 2), load_up=mu, load_down=md,
                            friction=(md - mu) / 2, bias=(mu + md) / 2,
                            lag_up=st.median(eu), lag_down=st.median(ed),
                            n_up=len(lu), n_down=len(ld)))
        a = b
    return out


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:] if not p.replace(".", "").isdigit()]
    width = next((float(p) for p in sys.argv[1:] if p.replace(".", "").isdigit()), 5.0)
    blocks: dict[str, list[dict]] = {}
    for p in paths:
        blocks.update(parse(p))
    stem = paths[0].with_suffix("") if len(paths) == 1 else Path(paths[0].parent, "full_range_2026-10-29")
    allbins = {j: bins_for(r, width) for j, r in blocks.items()}
    with open(f"{stem}_bins.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["joint", "center_deg", "load_up", "load_down", "friction", "bias", "lag_up_deg",
                    "lag_down_deg", "n_up", "n_down"])
        for j, bs in allbins.items():
            for b in bs:
                w.writerow([j, b["center"], b["load_up"], b["load_down"], b["friction"], b["bias"],
                            round(b["lag_up"], 2), round(b["lag_down"], 2), b["n_up"], b["n_down"]])
    for j, bs in allbins.items():
        rows = blocks[j]
        dt = [b["t"] - a["t"] for a, b in zip(rows, rows[1:])]
        aborted = rows and rows[-1]["phase"] not in ("back", "back_settle")
        print(f"\n{j}: {len(rows)} samples, {1 / st.median(dt):.0f} Hz"
              f"{'  (did not finish)' if aborted else ''}")
        if bs:
            fr = [b["friction"] for b in bs]
            print(f"  friction load median {st.median(fr):.0f}, range {min(fr):.0f}..{max(fr):.0f}; "
                  f"bias range {min(b['bias'] for b in bs):.0f}..{max(b['bias'] for b in bs):.0f}")
        print("  center  up   down  fric  bias  lag_up lag_dn")
        for b in bs:
            print(f"  {b['center']:6.1f} {b['load_up']:5.0f} {b['load_down']:5.0f} {b['friction']:5.0f} "
                  f"{b['bias']:5.0f}  {b['lag_up']:5.2f} {b['lag_down']:5.2f}")

    n = len(allbins)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.6 * n), squeeze=False)
    for (ax, ax2), (j, bs) in zip(axes, allbins.items()):
        rows = blocks[j]
        for ph, col in (("up", "#1f5fbf"), ("down", "#d0692a")):
            r = [x for x in rows if x["phase"] == ph]
            ax.plot([x["pos"] for x in r], [x["load"] for x in r], ".", ms=1, alpha=0.15, color=col)
        x = [b["center"] for b in bs]
        ax.plot(x, [b["load_up"] for b in bs], "-", color="#1f5fbf", lw=1.5, label="up (mean)")
        ax.plot(x, [b["load_down"] for b in bs], "-", color="#d0692a", lw=1.5, label="down (mean)")
        ax.axhline(0, color="#999", lw=0.5)
        ax.set_ylabel(f"{j}\nload")
        ax2.plot(x, [b["friction"] for b in bs], "-o", ms=3, color="#333", label="friction")
        ax2.plot(x, [b["bias"] for b in bs], "-o", ms=3, color="#2a9d5c", label="bias (gravity etc.)")
        ax2.axhline(0, color="#999", lw=0.5)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    axes[0, 0].set_title("load vs position, by direction")
    axes[0, 1].set_title(f"friction = (down-up)/2, bias = (up+down)/2, {width:g}° bins")
    axes[-1, 0].set_xlabel("joint angle (deg)")
    axes[-1, 1].set_xlabel("joint angle (deg)")
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
