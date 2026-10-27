#!/usr/bin/env python3
"""Checks of a log of reskin5x_mux.py (several boards behind an I2C mux): record rate, new conversions a second, zeros,
each chip's noise and the separation of the boards.

noise       the standard deviation of the difference of consecutive new conversions over sqrt 2 (uT, per axis), which
            leaves out drift; with --single a log of one board (wrench_viewer's stream log, 88-byte records) over the
            same kind of window, for comparison
separation  the correlation of those differences between every pair of components on different boards (a shared bus
            fault or a mux channel left open makes them correlate), and the records in which two boards report identical
            values for a chip (a missed channel write repeats the last board's values)
gaps        steps of the host time between consecutive records over 0.1 s (the splitter stamps a batch of records read
            together 1/150 s apart, back from the time of the read, so a stall of the USB stream shows as a gap)
malformed   records that the viewer drops (wrench_viewer.App.records: a chip whose field reads zero, a value that is not
            finite or over 1e5, a die temperature outside -40 to 125 C); the C6's stream of 2026-10-26 had single
            records whose bytes after the first 64 made no sense, though the record ended in CR LF. Noise and
            separation leave them out.

The log is read as a memory map, and only the window is loaded. --out writes the numbers as JSON.

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_mux_check.py LOG.bin [--boards 3]
        [--since 11:15 --until 11:20] [--single stream.bin --single-since 19:32 --single-until 19:38] [--out check.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CHIPS = 5


def clock(s, day):
    h, m, *sec = (int(v) for v in s.split(":"))
    lt = time.localtime(day)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, sec[0] if sec else 0, 0, 0, -1))


def window(path, n_boards, since=None, until=None, pad_s=0.0):
    """The records of a mux log between the clock times since and until (HH:MM[:SS] on the day of its first record,
    the next day when earlier than its start) widened by pad_s, as views of a memory map: host time t (n,) and the
    records r (n, boards, 5, 4) float32."""
    dt = np.dtype([("t", "<f8"), ("r", "<f4", (n_boards * CHIPS * 4,))])
    mm = np.memmap(path, dt, mode="r")
    t = np.array(mm["t"])
    i0, i1 = 0, len(t)
    if since:
        i0 = int(np.searchsorted(t, at(since, t[0]) - pad_s))
    if until:
        i1 = int(np.searchsorted(t, at(until, t[0]) + pad_s, side="right"))
    return t[i0:i1], mm["r"][i0:i1].reshape(i1 - i0, n_boards, CHIPS, 4)


def at(s, t_start):
    """The time of the clock reading s on the day of t_start, or on the next day when that falls before t_start (a log
    that runs past midnight)."""
    c = clock(s, t_start)
    return c + 86400 if c < t_start - 1 else c


def valid(r):
    """Mask (n,) or (n, boards) of the records (n, [boards,] 5, 4) that wrench_viewer.App.records keeps."""
    r = np.asarray(r)
    b, T = r[..., 1:], r[..., 0]
    with np.errstate(invalid="ignore", over="ignore"):
        ok = (~np.all(b == 0, axis=-1).any(axis=-1) & np.isfinite(r).all(axis=(-2, -1))
              & (np.abs(r) < 1e5).all(axis=(-2, -1)) & (T > -40).all(axis=-1) & (T < 125).all(axis=-1))
    return ok


def load(path, n_boards, since=None, until=None):
    t, r = window(path, n_boards, since, until)
    return t, r.astype(float)


def steps(v):
    """Differences of consecutive new values of one axis (n,) over sqrt 2, and the mask of records with a new value. In
    burst mode a chip's axes update one after another, so a record can carry a new x with the last y and z: each axis
    is taken on its own."""
    new = np.r_[True, np.diff(v) != 0]
    return np.diff(v[new]) / np.sqrt(2.0), new


def noise(r):
    return np.array([[steps(r[:, c, 1 + k])[0].std() for k in range(3)] for c in range(CHIPS)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log")
    ap.add_argument("--boards", type=int, default=3)
    ap.add_argument("--since", default=None)
    ap.add_argument("--until", default=None)
    ap.add_argument("--single", default=None)
    ap.add_argument("--single-since", default=None)
    ap.add_argument("--single-until", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t, r = load(a.log, a.boards, a.since, a.until)
    span = t[-1] - t[0]
    dt = np.diff(t)
    gaps = dt[dt > 0.1]
    ok = valid(r).all(axis=1)
    print(f"{len(t)} records over {span:.0f} s from {time.strftime('%H:%M:%S', time.localtime(t[0]))}: "
          f"{len(t) / span:.1f} records/s; {len(gaps)} gaps over 0.1 s (longest {dt.max():.3f} s); "
          f"{int((~ok).sum())} malformed records" + "".join(
              f" {time.strftime('%H:%M:%S', time.localtime(v))}" for v in t[~ok][:10]))
    out = dict(log=str(a.log), since=time.strftime("%H:%M:%S", time.localtime(t[0])),
               until=time.strftime("%H:%M:%S", time.localtime(t[-1])), records=int(len(t)), span_s=round(float(span), 1),
               rate=round(len(t) / span, 2), gaps_over_0p1s=int(len(gaps)), longest_step_s=round(float(dt.max()), 3),
               malformed=int((~ok).sum()), malformed_at=[time.strftime("%H:%M:%S", time.localtime(v)) for v in t[~ok]],
               boards=[])
    t, r = t[ok], r[ok]
    N = []
    for b in range(a.boards):
        rb = r[:, b]
        zeros = np.all(rb == 0, axis=2).sum(0)
        rate = [steps(rb[:, c, 3])[1].sum() / span for c in range(CHIPS)]
        nz = noise(rb)
        N.append(nz)
        print(f"board {b}: new z values/s {np.round(rate, 1)}, zero records {zeros}, die T "
              f"{np.round(rb[:, :, 0].mean(0), 1)} C")
        print(f"   noise (uT) x {np.round(nz[:, 0], 2)}, y {np.round(nz[:, 1], 2)}, z {np.round(nz[:, 2], 2)}")
        out["boards"].append(dict(new_z_per_s=np.round(rate, 1).tolist(), zero_records=zeros.tolist(),
                                  die_C=np.round(rb[:, :, 0].mean(0), 2).tolist(), noise_uT=np.round(nz, 3).tolist()))
    N = np.array(N)
    print(f"all boards: noise median x/y {np.median(N[:, :, :2]):.2f} uT, z {np.median(N[:, :, 2]):.2f} uT")
    out.update(noise_xy_uT=round(float(np.median(N[:, :, :2])), 3), noise_z_uT=round(float(np.median(N[:, :, 2])), 3))
    # separation: correlations of the differences of consecutive records (all components, records in common)
    D = np.diff(r[:, :, :, 1:].reshape(len(r), a.boards, CHIPS * 3), axis=0)
    flat = D.reshape(len(D), -1)
    keep = flat.std(0) > 0
    C = np.corrcoef(flat[:, keep].T)
    lab = np.repeat(np.arange(a.boards), CHIPS * 3)[keep]
    cross = np.abs(C[lab[:, None] != lab[None, :]])
    within = np.abs(C[(lab[:, None] == lab[None, :]) & ~np.eye(len(lab), dtype=bool)])
    print(f"separation: |correlation| between boards median {np.median(cross):.3f}, max {cross.max():.3f} "
          f"(within a board median {np.median(within):.3f}, max {within.max():.3f}); expected max for "
          f"{len(D)} independent samples about {4 / np.sqrt(len(D)):.3f}")
    out.update(corr_cross_median=round(float(np.median(cross)), 4), corr_cross_max=round(float(cross.max()), 4),
               corr_within_median=round(float(np.median(within)), 4), corr_bound=round(4 / np.sqrt(len(D)), 4))
    same = 0
    for b1 in range(a.boards):
        for b2 in range(b1 + 1, a.boards):
            same += int(np.sum(np.all(r[:, b1, :, 1:] == r[:, b2, :, 1:], axis=2) & (np.abs(r[:, b1, :, 1:]).sum(2) > 0)))
    print(f"chip readings identical on two boards: {same}")
    out["identical_readings"] = same
    if a.single:
        dt = np.dtype([("t", "<f8"), ("r", "<f4", (20,))])
        s = np.fromfile(a.single, dt)
        ts, rs = s["t"], s["r"].reshape(len(s), CHIPS, 4).astype(float)
        if a.single_since:
            m = (ts >= clock(a.single_since, ts[0])) & (ts <= clock(a.single_until, ts[0]))
            ts, rs = ts[m], rs[m]
        ns = noise(rs)
        rate = [steps(rs[:, c, 3])[1].sum() / (ts[-1] - ts[0]) for c in range(CHIPS)]
        print(f"single board ({len(ts)} records, {len(ts) / (ts[-1] - ts[0]):.1f}/s, new z values/s {np.round(rate, 1)}): "
              f"noise x {np.round(ns[:, 0], 2)}, y {np.round(ns[:, 1], 2)}, "
              f"z {np.round(ns[:, 2], 2)}; median x/y {np.median(ns[:, :2]):.2f}, z {np.median(ns[:, 2]):.2f} uT")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
