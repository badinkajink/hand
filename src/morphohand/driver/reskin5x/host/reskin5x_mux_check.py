#!/usr/bin/env python3
"""Checks of a log of reskin5x_mux.py (several boards behind an I2C mux): record rate, new conversions a second, zeros,
each chip's noise and the separation of the boards.

noise       the standard deviation of the difference of consecutive new conversions over sqrt 2 (uT, per axis), which
            leaves out drift; with --single a log of one board (wrench_viewer's stream log, 88-byte records) over the
            same kind of window, for comparison
separation  the correlation of those differences between every pair of components on different boards (a shared bus
            fault or a mux channel left open makes them correlate), and the records in which two boards report identical
            values for a chip (a missed channel write repeats the last board's values)

    ~/miniconda3/bin/python src/morphohand/driver/reskin5x/host/reskin5x_mux_check.py LOG.bin [--boards 3]
        [--since 11:15 --until 11:20] [--single stream.bin --single-since 19:32 --single-until 19:38]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

CHIPS = 5


def clock(s, day):
    h, m, *sec = (int(v) for v in s.split(":"))
    lt = time.localtime(day)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, sec[0] if sec else 0, 0, 0, -1))


def load(path, n_boards, since=None, until=None):
    dt = np.dtype([("t", "<f8"), ("r", "<f4", (n_boards * CHIPS * 4,))])
    a = np.fromfile(path, dt)
    t, r = a["t"], a["r"].reshape(len(a), n_boards, CHIPS, 4).astype(float)
    if since:
        m = t >= clock(since, t[0])
        if until:
            m &= t <= clock(until, t[0])
        t, r = t[m], r[m]
    return t, r


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
    a = ap.parse_args()
    t, r = load(a.log, a.boards, a.since, a.until)
    span = t[-1] - t[0]
    print(f"{len(t)} records over {span:.0f} s from {time.strftime('%H:%M:%S', time.localtime(t[0]))}: "
          f"{len(t) / span:.1f} records/s")
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
    N = np.array(N)
    print(f"all boards: noise median x/y {np.median(N[:, :, :2]):.2f} uT, z {np.median(N[:, :, 2]):.2f} uT")
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
    same = 0
    for b1 in range(a.boards):
        for b2 in range(b1 + 1, a.boards):
            same += int(np.sum(np.all(r[:, b1, :, 1:] == r[:, b2, :, 1:], axis=2) & (np.abs(r[:, b1, :, 1:]).sum(2) > 0)))
    print(f"chip readings identical on two boards: {same}")
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


if __name__ == "__main__":
    main()
