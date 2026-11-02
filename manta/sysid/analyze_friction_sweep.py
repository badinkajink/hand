#!/usr/bin/env python3
"""Analyze friction_sweep.py output: per joint and direction, breakaway load, tracking lag
while stuck, jump size, and the reversal dead band.

    python3 analyze_friction_sweep.py friction_sweep_raw_2026-10-29.txt

Writes <stem>.csv (all samples, with a joint column), <stem>_summary.csv and <stem>.png.
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COUNT = 300 / 1024  # deg per position count


def parse(path: Path) -> dict[str, list[dict]]:
    blocks: dict[str, list[dict]] = {}
    cur = None
    for line in path.read_text().splitlines():
        if line.startswith("CSV_BEGIN"):
            cur = line.split()[1]
            blocks[cur] = []
        elif line.startswith("CSV_END"):
            cur = None
        elif cur and line and not line.startswith("t_s"):
            t, ph, tg, p, ld = line.split(",")
            blocks[cur].append(dict(t=float(t), phase=ph, target=float(tg), pos=float(p),
                                    load=float(ld)))
    return blocks


def direction_stats(rows: list[dict], phase: str, sign: int) -> dict:
    """Stats for one ramp leg ('up' => +, 'down' => -), excluding its settle period."""
    r = [x for x in rows if x["phase"] == phase]
    if len(r) < 5:
        return {}
    jumps, breakaway = [], []
    for a, b in zip(r, r[1:]):
        dp = b["pos"] - a["pos"]
        if sign * dp >= 0.5 * COUNT:          # a slip in the commanded direction
            jumps.append(abs(dp))
            breakaway.append(a["load"])
    lag = [sign * (x["target"] - x["pos"]) for x in r]
    # Reversal dead band: target travel from the leg's start until the joint first moves
    # one count in the new direction.
    p0, tg0 = r[0]["pos"], r[0]["target"]
    dead = None
    for x in r:
        if sign * (x["pos"] - p0) >= 0.5 * COUNT:
            dead = abs(x["target"] - tg0)
            break
    travel = abs(r[-1]["target"] - tg0)
    moved = sign * (r[-1]["pos"] - p0)
    # Load while moving steadily: second half of the leg, past the reversal.
    half = r[len(r) // 2:]
    return dict(
        n_slips=len(jumps),
        breakaway_med=st.median(breakaway) if breakaway else None,
        breakaway_min=min(breakaway, key=abs) if breakaway else None,
        breakaway_max=max(breakaway, key=abs) if breakaway else None,
        jump_med=st.median(jumps) if jumps else None,
        jump_max=max(jumps) if jumps else None,
        lag_med=st.median(lag), lag_max=max(lag),
        dead_band=dead, travel=travel, moved=moved,
        load_med_2nd_half=st.median(x["load"] for x in half),
    )


def main() -> int:
    src = Path(sys.argv[1])
    blocks = parse(src)
    stem = src.with_suffix("")
    with open(f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["joint", "t_s", "phase", "target_deg", "pos_deg", "load"])
        for j, rows in blocks.items():
            for x in rows:
                w.writerow([j, x["t"], x["phase"], x["target"], x["pos"], x["load"]])

    summary = []
    for j, rows in blocks.items():
        dt = [b["t"] - a["t"] for a, b in zip(rows, rows[1:])]
        hz = 1 / st.median(dt) if dt else 0
        for phase, sign in (("up", +1), ("down", -1)):
            s = direction_stats(rows, phase, sign)
            if s:
                summary.append(dict(joint=j, dir=phase, hz=round(hz), **s))
    keys = list(summary[0].keys())
    with open(f"{stem}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for s in summary:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})

    fmt = lambda v, d=2: "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))  # noqa: E731
    print(f"{'joint':10} dir  Hz slips brk_med brk_rng      jump_med lag_med lag_max dead  moved/trav load2")
    for s in summary:
        rng = f"{fmt(s['breakaway_min'],0)}..{fmt(s['breakaway_max'],0)}"
        print(f"{s['joint']:10} {s['dir']:4} {s['hz']:3} {s['n_slips']:5} {fmt(s['breakaway_med'],0):>7} "
              f"{rng:12} {fmt(s['jump_med'])}     {fmt(s['lag_med'])}    {fmt(s['lag_max'])}    "
              f"{fmt(s['dead_band'])}  {fmt(s['moved'])}/{fmt(s['travel'])} {fmt(s['load_med_2nd_half'],0)}")

    n = len(blocks)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.4 * n), squeeze=False)
    for ax, (j, rows) in zip(axes[:, 0], blocks.items()):
        t = [x["t"] for x in rows]
        ax.plot(t, [x["target"] for x in rows], color="#888", lw=1, label="target")
        ax.step(t, [x["pos"] for x in rows], where="post", color="#1f5fbf", lw=1.2, label="position")
        ax.set_ylabel(f"{j}\ndeg")
        ax2 = ax.twinx()
        ax2.plot(t, [x["load"] for x in rows], color="#d0692a", lw=0.8, alpha=0.8, label="load")
        ax2.set_ylabel("load", color="#d0692a")
        ax2.axhline(0, color="#d0692a", lw=0.4, ls=":")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[-1, 0].set_xlabel("time in sweep (s)")
    fig.suptitle(f"Free-motion friction sweep, {src.stem}")
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
