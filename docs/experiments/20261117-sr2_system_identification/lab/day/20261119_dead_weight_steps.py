"""The owner's dead-weight check of the HEX-E (2026-11-19 17:4x): a weighed mass hung from the index fingertip's base
while the arm stays parked. Finds the steady stretches of the HEX-E's force (100 Hz, record_hexe.py's chunks) and the
steps between them; the step's magnitude |dF| (frame-free: the sensor's tilt only turns it) against m g.

    /usr/bin/python3 dead_weight_steps.py RECORD.jsonl --mass-g 205 [--win 3]
"""
import argparse
import json
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record")
    ap.add_argument("--mass-g", type=float, default=None)
    ap.add_argument("--win", type=float, default=3.0, help="s: the window of the rolling spread that marks steady")
    ap.add_argument("--steady-sd", type=float, default=0.12, help="N: x/y spread under which a window is steady")
    a = ap.parse_args()
    T, F = [], []
    for line in open(a.record):
        r = json.loads(line)
        if r["t"]:
            T += r["t"]
            F += r["F"]
    T, F = np.asarray(T), np.asarray(F)[:, :3]
    o = np.argsort(T)
    T, F = T[o], F[o]
    print(f"{len(T)} samples, {time.strftime('%H:%M:%S', time.localtime(T[0]))} to "
          f"{time.strftime('%H:%M:%S', time.localtime(T[-1]))}")
    # 1 s means, and which seconds are steady (x and y spread small; z is 4x noisier)
    secs = np.arange(np.floor(T[0]), np.ceil(T[-1]))
    rows = []
    for s in secs:
        m = (T >= s) & (T < s + 1)
        if m.sum() < 50:
            continue
        rows.append((s, F[m].mean(0), F[m].std(0)))
    S = np.array([r[0] for r in rows])
    M = np.array([r[1] for r in rows])
    SD = np.array([r[2] for r in rows])
    w = int(a.win)
    steady = np.array([i >= w and np.all(SD[i - w:i + 1, :2] < a.steady_sd) and
                       np.all(np.abs(M[i - w:i + 1] - M[i]).max(0)[:2] < 0.08) for i in range(len(S))])
    # group consecutive steady seconds into plateaus
    plats, cur = [], []
    for i in range(len(S)):
        if steady[i]:
            cur.append(i)
        elif cur:
            plats.append(cur)
            cur = []
    if cur:
        plats.append(cur)
    plats = [p for p in plats if len(p) >= 4]
    print("plateaus (start, end, seconds, mean F x y z):")
    P = []
    for p in plats:
        mu = M[p].mean(0)
        P.append((S[p[0]], S[p[-1]], mu))
        print(f"  {time.strftime('%H:%M:%S', time.localtime(S[p[0]]))}-{time.strftime('%H:%M:%S', time.localtime(S[p[-1]] + 1))} "
              f"{len(p):3d} s  F {np.round(mu, 3).tolist()}")
    g = 9.80665
    for (s0, e0, m0), (s1, e1, m1) in zip(P[:-1], P[1:]):
        d = m1 - m0
        line = (f"step {time.strftime('%H:%M:%S', time.localtime(e0))} -> {time.strftime('%H:%M:%S', time.localtime(s1))}: "
                f"dF {np.round(d, 3).tolist()} N, |dF| {np.linalg.norm(d):.3f} N")
        if a.mass_g:
            line += f"; m g {a.mass_g * 1e-3 * g:.3f} N -> HEX-E / true {np.linalg.norm(d) / (a.mass_g * 1e-3 * g):.3f}"
        print(line)


if __name__ == "__main__":
    main()
