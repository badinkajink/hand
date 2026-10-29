#!/usr/bin/env python3
"""The board's signal per newton on each region of the pad, for one or more analysed sessions (2026-10-29: whether the
tip's end, first pressed on v3 ID 2 in the low sphere's top-edge pass of 10-28 19:21, gives as much signal as the face).

The gain of a press hold is |d_board| / f_n: the norm of the board's change (its 15 components, uT) over the HEX-E's
normal force (N), as scripts/ur5_tip_calibration_four_vs_five.py defines it. Each group reports its median gain over the
holds with f_n in (0.5, 3.5] N and by the bins (0, 1], (1, 2], (2, 3.5] and (3.5, 8] N. Groups: the presses whose ids
start with --top (default "sph-top-", the top-edge pass) by region, without its reference presses; the grid presses of
the other plans by region; the reference presses at the face centre (FE index 788), those of the top-edge pass
apart. With --per-press the top-edge presses are also listed one by one with their gain at each level.

Input: the output folders of scripts/ur5_tip_calibration_analysis.py (holds.npz), as LABEL=DIR.

    .venv/bin/python scripts/ur5_tip_calibration_gain_by_region.py "v3 ID 2=logs/20261029-spherev3id2" \\
        "v3 ID 1=logs/20261028-spherev3id1_twists" [--per-press] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.5), (3.5, 8.0))
MID = (0.5, 3.5)


def in_bin(f, lo, hi):
    return (f > lo) & (f <= hi)


def groups(H, top):
    press = H["kind"] == "press"
    is_top = np.char.startswith(H["id"].astype(str), top)
    ref = H["role"] == "ref"
    g = {}
    for r in ("tip", "edge", "flank", "face"):
        m = press & is_top & ~ref & (H["region"] == r)
        if m.any():
            g[f"top-edge pass, {r}"] = m
    for r in ("face", "edge", "flank", "tip"):
        m = press & ~is_top & ~ref & (H["stage"] == "grid") & (H["region"] == r)
        if m.any():
            g[f"grid, {r}"] = m
    for name, m in (("reference presses of the top-edge pass", press & ref & is_top),
                    ("reference presses, other plans", press & ref & ~is_top)):
        if m.any():
            g[name] = m
    return g


def stats(gain, f, m):
    row = dict(holds=int(m.sum()))
    mm = m & in_bin(f, *MID)
    row["gain_0.5-3.5"] = float(np.median(gain[mm])) if mm.any() else None
    for lo, hi in BINS:
        mb = m & in_bin(f, lo, hi)
        row[f"gain_{lo:g}-{hi:g}"] = float(np.median(gain[mb])) if mb.any() else None
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sessions", nargs="+", help="LABEL=DIR, DIR an analysis output with holds.npz")
    ap.add_argument("--top", default="sph-top-")
    ap.add_argument("--per-press", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = dict(script="scripts/ur5_tip_calibration_gain_by_region.py", gain="median |d_board| / f_n (uT/N)", sessions={})
    for s in a.sessions:
        label, d = s.split("=", 1)
        H = np.load(Path(d) / "holds.npz")
        f = H["f_n"]
        gain = np.linalg.norm(H["d_board"], axis=1) / np.where(f > 0, f, np.nan)
        res = dict(holds=d, groups={k: stats(gain, f, m) for k, m in groups(H, a.top).items()})
        print(f"{label} ({d}):")
        for k, v in res["groups"].items():
            bins = "  ".join(f"{lo:g}-{hi:g} " + (f"{v[f'gain_{lo:g}-{hi:g}']:5.1f}" if v[f"gain_{lo:g}-{hi:g}"] is not None
                                                  else "    -") for lo, hi in BINS)
            mid = f"{v['gain_0.5-3.5']:5.1f}" if v["gain_0.5-3.5"] is not None else "    -"
            print(f"  {k:42s} {v['holds']:5d} holds  0.5-3.5 N {mid} uT/N   {bins}")
        if a.per_press:
            rows = []
            m = (H["kind"] == "press") & np.char.startswith(H["id"].astype(str), a.top)
            for i in dict.fromkeys(H["id"][m].astype(str)):
                k = m & (H["id"] == i)
                rows.append(dict(id=i, region=str(H["region"][k][0]), role=str(H["role"][k][0]),
                                 levels=[dict(f_n=round(float(x), 3), gain=round(float(y), 2), d_board_uT=round(float(z), 2))
                                         for x, y, z in zip(f[k], gain[k], np.linalg.norm(H["d_board"][k], axis=1))]))
                print(f"    {i:24s} {rows[-1]['region']:5s} " + "  ".join(f"{x['f_n']:4.2f} N {x['gain']:5.1f}"
                                                                          for x in rows[-1]["levels"]))
            res["per_press"] = rows
        out["sessions"][label] = res
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
