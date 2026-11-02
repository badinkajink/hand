#!/usr/bin/env python3
"""Compare two full-range sweep passes: does each joint's friction-vs-angle profile repeat,
and where are the sticking spots?

    python3 compare_full_range_passes.py --a A1.txt [A2.txt ...] --b B1.txt [B2.txt ...] [--bin 2]

Pass A / pass B files are full_range_sweep.py raw outputs (later files override earlier ones for
the same joint). A bin is a sticking spot when friction exceeds SPOT_X x that joint's median in
BOTH passes. Writes full_range_passes_2026-10-29.{csv,png}.
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze_friction_sweep import parse  # noqa: E402
from analyze_full_range import bins_for  # noqa: E402

SPOT_X = 1.5


def load(paths: list[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in paths:
        out.update(parse(Path(p)))
    return out


def corr(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    mx, my = st.fmean(x), st.fmean(y)
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--bin", type=float, default=2.0)
    ap.add_argument("--out", default="full_range_passes_2026-10-29")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    joints = [j for j in A if j in B]
    rows_out = []
    fig, axes = plt.subplots(len(joints), 1, figsize=(10, 2.3 * len(joints)), squeeze=False)
    print(f"{'joint':8} {'bins':>4} {'medA':>5} {'medB':>5} {'r':>5}  sticking spots (friction > {SPOT_X}x median, both passes)")
    for ax, j in zip(axes[:, 0], joints):
        ba = {b["center"]: b for b in bins_for(A[j], args.bin)}
        bb = {b["center"]: b for b in bins_for(B[j], args.bin)}
        # pair bins by nearest center (pass ranges can start a little differently)
        pairs = []
        for ca, xa in ba.items():
            cb = min(bb, key=lambda c: abs(c - ca), default=None)
            if cb is not None and abs(cb - ca) <= args.bin / 2:
                pairs.append((ca, xa, bb[cb]))
        fa = [p[1]["friction"] for p in pairs]
        fb = [p[2]["friction"] for p in pairs]
        if not pairs:
            print(f"{j:8} no overlapping bins")
            continue
        ma, mb = st.median(fa), st.median(fb)
        spots = [p[0] for p in pairs
                 if p[1]["friction"] > SPOT_X * ma and p[2]["friction"] > SPOT_X * mb]
        # merge adjacent spot bins into ranges
        ranges: list[list[float]] = []
        for c in spots:
            if ranges and c - ranges[-1][1] <= args.bin * 1.01:
                ranges[-1][1] = c
            else:
                ranges.append([c, c])
        txt = ", ".join(f"{a - args.bin / 2:.0f}..{b + args.bin / 2:.0f}°" for a, b in ranges) or "none"
        print(f"{j:8} {len(pairs):4} {ma:5.0f} {mb:5.0f} {corr(fa, fb):5.2f}  {txt}")
        for c, xa, xb in pairs:
            rows_out.append([j, c, round(xa["friction"], 1), round(xb["friction"], 1),
                             round(xa["bias"], 1), round(xb["bias"], 1), int(c in spots)])
        ax.plot([p[0] for p in pairs], fa, "-o", ms=2, color="#1f5fbf", label="pass 1")
        ax.plot([p[0] for p in pairs], fb, "-o", ms=2, color="#d0692a", label="pass 2")
        for a, b in ranges:
            ax.axvspan(a - args.bin / 2, b + args.bin / 2, color="#e8c547", alpha=0.35, lw=0)
        ax.set_ylabel(f"{j}\nfriction")
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title(f"Friction vs angle, two passes ({args.bin:g}° bins); yellow = sticking spot in both")
    axes[-1, 0].set_xlabel("joint angle (deg)")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=110)
    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["joint", "center_deg", "friction_pass1", "friction_pass2", "bias_pass1",
                    "bias_pass2", "sticking_spot"])
        w.writerows(rows_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
