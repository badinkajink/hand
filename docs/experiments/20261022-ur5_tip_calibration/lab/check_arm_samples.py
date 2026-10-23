"""Checks every robot sample of a session (robot.bin, real or simulated) against the arm rules of 2026-10-22: joint 5
inside the wrist window and the HEX-E, its cable port, the mount and the finger clear of the robot's own forearm and
wrist links (scripts/ur5_cb3_cell.py self_clearance). Prints each stage's joint 5 range and least clearances and every
sample that breaks a rule. Moves nothing.

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/check_arm_samples.py SESSION [--every 5]
        [--window 30,100] [--turn -92.15]
"""
import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_cb3_cell as CELL  # noqa: E402

C = CELL.C
ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--every", type=int, default=5, help="check every n-th sample (125 Hz stream)")
ap.add_argument("--window", default="30,100")
ap.add_argument("--turn", type=float, default=-92.15, help="the mount's turn about the tool axis, deg")
a = ap.parse_args()
sd = Path(a.session)
lo, hi = (float(v) for v in a.window.split(","))
r = np.fromfile(sd / "robot.bin", "<f8").reshape(-1, 15)
t, q = r[:, 0], r[:, 2:8]
ev = [json.loads(s) for s in open(sd / "events.jsonl")]
starts = [(e["t"], e.get("stage") or e.get("id")) for e in ev if e.get("event") == "start"]
T_EM = C.Tm(C.rot(C.EZ, a.turn), [0.0, 0.0, 35.0])


def stage_at(x):
    st = "start"
    for ts, s in starts:
        if ts <= x:
            st = s
    return st


ts = (lambda x: datetime.datetime.fromtimestamp(x).strftime("%H:%M:%S")) if t[0] > 1e9 else (lambda x: f"{x:9.1f} s")
per, bad = {}, []
for i in range(0, len(t), a.every):
    st = stage_at(t[i])
    q5 = math.degrees(q[i, 4])
    sc = CELL.self_clearance(q[i], T_EM)
    d = per.setdefault(st, dict(q5=[1e9, -1e9], port=1e9, hex_e=1e9, tool=1e9, n=0))
    d["q5"] = [min(d["q5"][0], q5), max(d["q5"][1], q5)]
    for k in ("port", "hex_e", "tool"):
        d[k] = min(d[k], sc[k])
    d["n"] += 1
    if not (lo <= q5 <= hi) or any(sc[k] < CELL.SELF_CLEAR[k] for k in CELL.SELF_CLEAR):
        bad.append((ts(t[i]), st, round(q5, 1), {k: round(v, 1) for k, v in sc.items()}))
for st, d in per.items():
    print(f"{st:14s} {d['n']:6d} samples  joint 5 {d['q5'][0]:6.1f} to {d['q5'][1]:6.1f} deg  port >= {d['port']:5.1f}  "
          f"HEX-E >= {d['hex_e']:5.1f}  mount and finger >= {d['tool']:5.1f} mm")
print(f"{len(bad)} samples break a rule (window {lo:.0f} to {hi:.0f} deg, clearances {CELL.SELF_CLEAR})")
for b in bad[:20]:
    print("  ", b)
