#!/usr/bin/env python3
"""Compare double-pendulum recordings across servo gains (P, D): tracking error, jitter /
oscillation, full-PWM saturation. Real data only (no model).

    python3 compare_gains.py data_raw/double_v2 data_raw/double_v2_highkp data_raw/double_v2_kd15 ...

Per (P, D), averaged over paths, during the "path" phase:
  rms      RMS of goal - position (deg)
  jitter   RMS of position minus its own 0.25 s moving average (deg): oscillation / stick-slip
           that doesn't follow the goal. The goal paths are smooth on this scale except the
           square-wave steps, so square_wave is left out of jitter.
  rev/s    direction reversals of position per second
  sat%     share of samples with |load| >= 990 (servo at full PWM)
"""
from __future__ import annotations

import glob
import json
import math
import statistics as st
import sys


def hp_rms(t: list[float], x: list[float], win: float = 0.25) -> float:
    out, j0, s = [], 0, 0.0
    # centered moving average via a two-pointer window
    n = len(x)
    lo = hi = 0
    acc = 0.0
    for i in range(n):
        while hi < n and t[hi] <= t[i] + win / 2:
            acc += x[hi]
            hi += 1
        while t[lo] < t[i] - win / 2:
            acc -= x[lo]
            lo += 1
        out.append(x[i] - acc / (hi - lo))
    return math.sqrt(st.fmean(v * v for v in out))


def reversals(t: list[float], x: list[float]) -> float:
    n, last = 0, 0
    for a, b in zip(x, x[1:]):
        if b != a:
            d = 1 if b > a else -1
            if last and d != last:
                n += 1
            last = d
    return n / (t[-1] - t[0])


def main() -> int:
    groups: dict[tuple[int, int], list[dict]] = {}
    for d in sys.argv[1:]:
        for f in sorted(glob.glob(f"{d}/*.json")):
            log = json.load(open(f))
            if log.get("status", "ok") != "ok":
                continue
            e = [x for x in log["entries"] if x["phase"] == "path"]
            t = [x["timestamp"] for x in e]
            r = {"path": log["path"]}
            for j in range(2):
                pos = [x["position"][j] for x in e]
                r[f"rms{j}"] = math.sqrt(st.fmean((x["goal_position"][j] - x["position"][j]) ** 2 for x in e))
                r[f"jit{j}"] = hp_rms(t, pos) if log["path"] != "square_wave" else None
                r[f"rev{j}"] = reversals(t, pos)
                r[f"sat{j}"] = 100 * st.fmean(abs(x["load"][j]) >= 990 for x in e)
            groups.setdefault((log["kp"], log.get("kd", 15)), []).append(r)
    print(f"{'P':>3} {'D':>3} {'n':>2} | {'rms j1':>6} {'rms j2':>6} | {'jit j1':>6} {'jit j2':>6} | "
          f"{'rev/s j1':>8} {'rev/s j2':>8} | {'sat% j1':>7} {'sat% j2':>7}")
    for (kp, kd), rs in sorted(groups.items(), key=lambda k: (k[0][1], k[0][0])):
        def m(k):
            v = [r[k] for r in rs if r[k] is not None]
            return st.fmean(v) if v else float("nan")
        print(f"{kp:3d} {kd:3d} {len(rs):2d} | {m('rms0'):6.2f} {m('rms1'):6.2f} | {m('jit0'):6.2f} {m('jit1'):6.2f} | "
              f"{m('rev0'):8.1f} {m('rev1'):8.1f} | {m('sat0'):7.1f} {m('sat1'):7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
