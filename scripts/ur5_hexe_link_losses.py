#!/usr/bin/env python3
"""The HEX-E's link losses on the UR5 against the arm's joints (2026-10-30).

Reads local copies of corvus's session logs, one folder per session (logs/<date>-tip_calibration/<HHMMSS>-<name> on
corvus): robot.bin (host time, controller time, q0..q5 in rad, flange pose, safety mode; 125 Hz) and ft.bin (host time
of each HEX-E sample, seq, status, force and torque; 100 Hz). A loss is a gap of more than GAP_S in the HEX-E's samples
while robot.bin keeps running; the joints at the last sample before it are the loss pose, and the abort in the lab
folder's events.jsonl that follows it names the action. Then, over all sessions:

- visits: each stretch of time the arm spends within --tol deg of a loss pose on all six joints, and whether a loss
  ends it (within 2 s of its end);
- travels: each motion whose wrist joints move more than 10 deg, classed as scripts/ur5_tip_calibration_wrist_order.py
  classes the replayed ones (the losses' signature: wrist 1 from below 0 to above it while wrist 2 rises more than
  10 deg; a large swing: wrist 1 crosses 0 and moves more than 40 deg), and whether a loss comes during it or in the
  10 s after it.

    python3 scripts/ur5_hexe_link_losses.py logs/20261030-hexe_losses --out OUT.json [--tol 3] [--fig OUT.png]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[1] / "docs/experiments/20261022-ur5_tip_calibration/lab"
GAP_S = 1.0              # a longer gap in the HEX-E's samples, with robot.bin running through it, is a loss
ALIVE_S = 0.2            # the link is alive at a robot sample when a HEX-E sample lies within this
MOVING = 0.8             # deg/s on the fastest joint; the tall plans travel at about 2 deg/s
NAMES = ["base", "shoulder", "elbow", "wrist 1", "wrist 2", "wrist 3"]


def hms(t):
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]


def segments(mask, t, join_s):
    """(first, last) row indices of the runs of True in mask, joining runs less than join_s apart."""
    i = np.flatnonzero(mask)
    if not len(i):
        return []
    cut = np.flatnonzero(np.diff(t[i]) > join_s)
    return [(int(i[a]), int(i[b])) for a, b in zip(np.r_[0, cut + 1], np.r_[cut, len(i) - 1])]


def read(path, cols):
    with open(path, "rb") as f:
        x = np.fromfile(f, "<f8").reshape(-1, cols)
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)      # the page cache counts against the cgroup
    return x


def load(sd):
    """At 25 Hz: host time, joints (deg), the link alive, the fastest joint's speed (deg/s); and the HEX-E's times."""
    R = read(sd / "robot.bin", 15)
    t, q = R[::5, 0].copy(), np.degrees(R[::5, 2:8])
    del R
    tF = read(sd / "ft.bin", 9)[:, 0].copy()
    k = np.clip(np.searchsorted(tF, t), 1, len(tF) - 1)
    alive = np.minimum(np.abs(tF[k] - t), np.abs(tF[k - 1] - t)) < ALIVE_S
    a, b = np.clip(np.arange(len(t)) - 3, 0, len(t) - 1), np.clip(np.arange(len(t)) + 3, 0, len(t) - 1)
    speed = np.abs(q[b] - q[a]).max(1) / np.maximum(t[b] - t[a], 1e-3)
    return t, q, tF, alive, speed


def find_losses(name, t, q, tF, speed):
    starts = list(tF[:-1][np.diff(tF) > GAP_S]) + ([tF[-1]] if t[-1] - tF[-1] > GAP_S else [])
    aborts = []
    for f in LAB.glob(f"*-{name}/events.jsonl"):
        aborts += [e for e in map(json.loads, f.open()) if e["event"] == "end" and e.get("status") == "abort"]
    out = []
    for t0 in starts:
        k = np.searchsorted(tF, t0, side="right")
        t1 = tF[k] if k < len(tF) else t[-1]
        if np.count_nonzero((t > t0) & (t < t1)) < 12:      # the robot's stream stopped too: the run ended
            continue
        j = min(int(np.searchsorted(t, t0)), len(t) - 1)
        moved = np.flatnonzero(speed[:j + 1] > MOVING)
        ab = [e for e in aborts if t0 <= e["t"] <= t0 + 30.0]
        out.append(dict(session=name, time=hms(t0), t=float(t0), gap_s=round(float(t1 - t0), 2) if k < len(tF) else None,
                        q_deg=[round(float(x), 1) for x in q[j]], speed_deg_s=round(float(speed[j]), 1),
                        still_s=round(float(t0 - t[moved[-1]]), 1) if len(moved) else None,
                        abort=ab[0]["id"] if ab else None, why=ab[0].get("why") if ab else None))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", help="a folder of session folders, each with robot.bin and ft.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=3.0, help="deg on every joint around a loss pose")
    ap.add_argument("--fig")
    a = ap.parse_args()
    data, losses = {}, []
    for sd in sorted(p for p in Path(a.logs).iterdir() if (p / "robot.bin").exists() and (p / "ft.bin").exists()):
        t, q, tF, alive, speed = load(sd)
        losses += find_losses(sd.name, t, q, tF, speed)
        data[sd.name] = (t, q, alive, speed)
    losses.sort(key=lambda L: L["t"])
    t_loss = np.array([L["t"] for L in losses])

    hours = {n: round(float(np.clip(np.diff(d[0]), 0, 0.5).sum() / 3600), 2) for n, d in data.items()}
    visits = []
    for L in losses:
        pose, rows = np.array(L["q_deg"]), []
        for n, (t, q, alive, speed) in data.items():
            near = (np.abs(q - pose) <= a.tol).all(1)
            for i, j in segments(near, t, 2.0):
                lost = bool(((t_loss >= t[i] - 0.5) & (t_loss <= t[j] + 2.0)).any())
                rows.append(dict(session=n, start=hms(t[i]), s=round(float(t[j] - t[i]), 1),
                                 alive_s=round(float(np.clip(np.diff(t[i:j + 1]), 0, 0.5)[alive[i:j]].sum()), 1),
                                 lost=lost))
        visits.append(dict(loss=L["time"], pose=L["q_deg"], n=len(rows), lost=sum(r["lost"] for r in rows),
                           alive_s=round(sum(r["alive_s"] for r in rows), 1), visits=rows))

    travels = []
    for n, (t, q, alive, speed) in data.items():
        segs = [(i, j) for i, j in segments(speed > MOVING, t, 1.0)
                if np.abs(q[j, 3:] - q[i, 3:]).sum() >= 10.0 or np.ptp(q[i:j + 1, 3]) >= 10.0]
        for m, (i, j) in enumerate(segs):
            w = q[i:j + 1]
            k = int(np.argmax(w[:, 3]))
            t_end = min(t[j] + 10.0, t[segs[m + 1][0]] if m + 1 < len(segs) else np.inf)   # until the next travel
            travels.append(dict(session=n, start=hms(t[i]), q0=[round(float(x), 1) for x in w[0]],
                                q1=[round(float(x), 1) for x in w[-1]],
                                signature=bool(w[0, 3] < 0.0 < w[k, 3] and w[k, 4] - w[0, 4] > 10.0),
                                large_swing=bool(w[:, 3].min() < 0.0 < w[:, 3].max() and abs(w[-1, 3] - w[0, 3]) > 40.0),
                                lost=bool(((t_loss >= t[i]) & (t_loss <= t_end)).any())))

    def count(sel):
        s = [r for r in travels if sel(r)]
        return dict(n=len(s), lost=sum(r["lost"] for r in s))
    summary = dict(signature=count(lambda r: r["signature"]), large_swing=count(lambda r: r["large_swing"]),
                   other=count(lambda r: not r["signature"] and not r["large_swing"]))
    out = dict(gap_s=GAP_S, tol_deg=a.tol, hours=hours, losses=losses, travels=summary, visits=visits,
               travel_rows=travels)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")

    for L in losses:
        print(f"{L['time'][5:]}  {L['session']:20s} q " + " ".join(f"{x:6.1f}" for x in L["q_deg"]) +
              f"  {L['speed_deg_s']:4.1f} deg/s  still {L['still_s']} s  gap {L['gap_s']} s  -> {L['abort']}")
    for v in visits:
        print(f"{v['loss'][5:]}: {v['n']} visits within {a.tol:g} deg, {v['lost']} ended by a loss, "
              f"{v['alive_s']} s alive there")
    print("travels:", json.dumps(summary))
    print(f"{sum(hours.values()):.1f} h of logs in {len(hours)} sessions ->", a.out)

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), dpi=130)
        for k, (x, y) in enumerate([(3, 4), (3, 5)]):
            for n, (t, q, alive, speed) in data.items():
                s = np.flatnonzero(alive)[::25]                 # one point per second
                ax[k].scatter(q[s, x], q[s, y], s=2, c="0.75", rasterized=True)
            P = np.array([L["q_deg"] for L in losses])
            ax[k].scatter(P[:, x], P[:, y], s=46, c="tab:red", marker="x", label="HEX-E link lost")
            ax[k].set_xlabel(f"{NAMES[x]} (deg)", fontsize=13)
            ax[k].set_ylabel(f"{NAMES[y]} (deg)", fontsize=13)
            ax[k].tick_params(labelsize=12)
        ax[0].scatter([], [], s=8, c="0.75", label="link alive, one point per second")
        ax[0].legend(fontsize=12, loc="lower right")
        fig.tight_layout()
        fig.savefig(a.fig)
        print("->", a.fig)


if __name__ == "__main__":
    main()
