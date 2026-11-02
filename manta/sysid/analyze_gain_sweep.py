#!/usr/bin/env python3
"""Summarize gain_sweep.py output per setting: tracking error, slip count and size, holding
error and load. Writes <stem>_summary.csv and <stem>.png.

    python3 analyze_gain_sweep.py gain_sweep_raw_<...>.txt
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COUNT = 300 / 1024


def main() -> int:
    src = Path(sys.argv[1])
    lines = src.read_text().split("CSV_BEGIN gains\n", 1)[1].split("CSV_END")[0].splitlines()[1:]
    runs: list[dict] = []          # in order; a setting can appear twice
    for ln in lines:
        t, tag, ph, tg, p, v = ln.split(",")
        if not runs or runs[-1]["tag"] != tag or runs[-1].get("closed"):
            runs.append({"tag": tag, "sine": [], "hold": [], "temp": None})
        r = runs[-1]
        if ph == "temp":
            r["temp"] = float(v)
            r["closed"] = True
        else:
            r[ph].append((float(t), float(tg), float(p), float(v)))
    out = []
    for k, r in enumerate(runs):
        sine, hold = r["sine"], r["hold"]
        dur = sine[-1][0] - sine[0][0]
        err = [p - tg for _t, tg, p, _l in sine]
        jumps = [abs(b[2] - a[2]) for a, b in zip(sine, sine[1:]) if b[2] != a[2]]
        # time spent stuck: longest run with no position change
        stuck, longest, start = [], 0.0, sine[0][0]
        for a, b in zip(sine, sine[1:]):
            if b[2] != a[2]:
                stuck.append(b[0] - start)
                start = b[0]
        hold_err = [abs(p - tg) for _t, tg, p, _l in hold[len(hold) // 2:]]
        hold_load = [abs(l) for *_x, l in hold[len(hold) // 2:]]
        # jitter: RMS of position minus its own 0.25 s centered moving average (same measure as
        # bam_pendulum/compare_gains.py), with single-sample read tears (> 1 deg away from both
        # neighbours) dropped first
        clean = [b for a, b, c in zip(sine, sine[1:], sine[2:])
                 if not (abs(b[2] - a[2]) > 1 and abs(b[2] - c[2]) > 1)]
        ts, ps = [x[0] for x in clean], [x[2] for x in clean]
        hp, lo_i, hi_i, acc = [], 0, 0, 0.0
        for i in range(len(ps)):
            while hi_i < len(ps) and ts[hi_i] <= ts[i] + 0.125:
                acc += ps[hi_i]
                hi_i += 1
            while ts[lo_i] < ts[i] - 0.125:
                acc -= ps[lo_i]
                lo_i += 1
            hp.append(ps[i] - acc / (hi_i - lo_i))
        out.append(dict(
            run=k + 1, setting=r["tag"], hz=round(len(sine) / dur),
            rms_err=round((st.fmean(e * e for e in err)) ** 0.5, 2),
            jitter=round(st.fmean(v * v for v in hp) ** 0.5, 3),
            max_err=round(max(abs(e) for e in err), 2),
            moves_per_s=round(len(jumps) / dur, 1),
            jump_med=round(st.median(jumps), 2) if jumps else None,
            jump_max=round(max(jumps), 2) if jumps else None,
            stuck_med_ms=round(st.median(stuck) * 1e3) if stuck else None,
            load_abs_mean=round(st.fmean(abs(l) for *_x, l in sine)),
            hold_err=round(st.median(hold_err), 2), hold_load=round(st.median(hold_load)),
            temp_c=r["temp"]))
    keys = list(out[0])
    with open(f"{src.with_suffix('')}_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(out)
    print(" ".join(f"{k:>12}" for k in keys))
    for o in out:
        print(" ".join(f"{str(o[k]):>12}" for k in keys))
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 1.9 * len(runs)), sharex=True)
    for ax, r, o in zip(axes, runs, out):
        t0 = r["sine"][0][0]
        sel = [x for x in r["sine"] if x[0] - t0 < 4.0]
        ax.plot([x[0] - t0 for x in sel], [x[1] for x in sel], color="#888", lw=1)
        ax.step([x[0] - t0 for x in sel], [x[2] for x in sel], where="post", color="#1f5fbf", lw=1.2)
        ax.set_ylabel(f"{o['setting']}\n#{o['run']}", fontsize=8)
        ax.text(0.99, 0.05, f"rms {o['rms_err']}°, {o['moves_per_s']} moves/s, "
                f"jump {o['jump_med']}°", transform=ax.transAxes, ha="right", fontsize=8)
    axes[-1].set_xlabel("time in sine (s), first 4 s; grey = target, blue = position")
    fig.suptitle(f"Gain sweep, {src.stem}")
    fig.tight_layout()
    fig.savefig(f"{src.with_suffix('')}.png", dpi=110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
