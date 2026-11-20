#!/usr/bin/env python3
"""Where a bench series' time goes, from hom_series's logs (hand-4a, 2026-11-20 12:1x; owner, 11:4x: "on the ur5,
let's continually look for procedures/common events that can automated/added to a decision tree/accelerated ... we
really want to maximize real world throughput"; the lab asked for this table at 11:5x).

A trial's block in the log runs from its "=== HH:MM:SS trial n/N: label" header to the next header. Its phases, from
the times the log prints:
    setup     header to the fingertip splitter's attach: the arm's clear, layout, place and approach moves and ready
    rest      the attach to the trial's start: the fingertip boards' rest readings and re-zero
    trial     the trial's start to "trial done": HOM's twist with the arm's lift
    putdown   "trial done" to the next trial's header: putdown, letgo, recentring
    stopped   "trial done" to the next "restart" header, for a block that ended in STOP: the putdown attempts and the
              recovery by hand, until the series was restarted
A block without a trial start (a setup that failed) counts as setup up to its STOP or the next header.

    python3 docs/experiments/20261115-hom_bench_trials/analysis/20261120-series_time_table.py LOG [LOG ...]
Prints the table and writes 20261120-series_time_table.json next to this script.
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HDR = re.compile(r"^=== (\d\d:\d\d:\d\d) (trial \d+/\d+: (\S+)|restart.*)")
TS = re.compile(r"^\s+(\d\d:\d\d:\d\d) (fingertips: splitter attached|trial twist:|trial done)")


def sec(hms):
    h, m, s = (int(v) for v in hms.split(":"))
    return 3600 * h + 60 * m + s


def parse(path):
    """[{label, t0, attach, start, done, end, stop}] of one log, times in s of the day."""
    out, cur = [], None
    for line in Path(path).read_text(errors="replace").splitlines():
        m = HDR.match(line)
        if m:
            t = sec(m.group(1))
            if cur is not None:
                cur["end"] = t
                cur["end_kind"] = "restart" if m.group(2).startswith("restart") else "next"
                out.append(cur)
                cur = None
            if m.group(3):
                cur = {"label": m.group(3), "t0": t, "attach": None, "start": None, "done": None, "stop": None}
            continue
        if cur is None:
            continue
        m = TS.match(line)
        if m:
            key = {"fingertips: splitter attached": "attach", "trial twist:": "start", "trial done": "done"}[m.group(2)]
            cur[key] = sec(m.group(1))
        elif line.startswith("STOP:"):
            cur["stop"] = line[5:].strip()[:160]
    if cur is not None:
        cur["end"], cur["end_kind"] = None, "open"
        out.append(cur)
    return out


def phases(b):
    """The block's phase durations (s); None where the log has no time for it."""
    t0, at, st, dn, en = b["t0"], b["attach"], b["start"], b["done"], b["end"]
    ph = {"setup": None, "rest": None, "trial": None, "putdown": None, "stopped": None}
    last = dn or st or at or t0
    ph["setup"] = (at if at else (en if en and not st else None)) and ((at or en) - t0)
    if at and st:
        ph["rest"] = st - at
    if st and dn:
        ph["trial"] = dn - st
    if en is not None:
        if b["stop"] or b["end_kind"] == "restart":
            ph["stopped"] = en - last
        elif dn:
            ph["putdown"] = en - dn
    return ph


def main():
    logs = sys.argv[1:]
    blocks = [b for p in logs for b in parse(p)]
    rows = []
    for b in blocks:
        rows.append({"label": b["label"], "header": time.strftime("%H:%M:%S", time.gmtime(b["t0"])),
                     "stop": b["stop"], **phases(b)})
    keys = ("setup", "rest", "trial", "putdown", "stopped")
    tot = {k: float(np.nansum([r[k] for r in rows if r[k] is not None])) for k in keys}
    span = sum(tot.values())
    print(f"{len(rows)} trial blocks in {len(logs)} logs; {span / 60:.1f} min accounted")
    print(f"{'phase':9s} {'trials':>6s} {'median s':>9s} {'total min':>9s} {'share':>6s}")
    summ = {}
    for k in keys:
        v = [r[k] for r in rows if r[k] is not None]
        summ[k] = {"n": len(v), "median_s": round(float(np.median(v)), 1) if v else None,
                   "total_min": round(tot[k] / 60, 1), "share": round(tot[k] / span, 3) if span else None}
        print(f"{k:9s} {len(v):6d} {summ[k]['median_s'] if v else float('nan'):9.1f} {tot[k] / 60:9.1f} "
              f"{100 * tot[k] / span if span else 0:5.1f}%")
    stops = [r for r in rows if r["stop"]]
    for r in stops:
        print(f"  stop at {r['label']}: {r['stopped'] or 0:.0f} s; {r['stop'][:110]}")
    full = [r for r in rows if r["putdown"] is not None and r["setup"] is not None and r["trial"] is not None]
    if full:
        cyc = [r["setup"] + (r["rest"] or 0) + r["trial"] + r["putdown"] for r in full]
        print(f"trial cycle without stops: median {np.median(cyc):.0f} s over {len(full)} trials "
              f"({3600 / np.median(cyc):.1f} trials per hour)")
    out = HERE / "20261120-series_time_table.json"
    out.write_text(json.dumps({"what": __doc__.split("\n\n")[0].replace("\n", " "), "logs": [Path(p).name for p in logs],
                               "generated": time.strftime("%F %H:%M"), "summary": summ, "rows": rows}, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
