"""Each finished press of SESSION: the flange's travel from its rest to the poses at its 0.5 N and 3 N levels (the
level's first time), along the press, from robot.bin, and the 0.5 -> 3 N travel. On the pad the rise takes about
0.5-0.7 mm; on the tip's rigid body much less. IDS: only these (default all). Reads files only.

    /usr/bin/python3 docs/experiments/20261022-ur5_tip_calibration/lab/press_depth.py SESSION [ID ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

S = Path(sys.argv[1])
want = set(sys.argv[2:])
r = np.fromfile(S / "robot.bin", dtype="<f8")
r = r[: r.size // 15 * 15].reshape(-1, 15)
t, p = r[:, 0], r[:, 8:11] * 1000.0


def at(tt):
    return np.array([np.interp(tt, t, p[:, k]) for k in range(3)])


rows = []
for line in open(S / "events.jsonl"):
    if '"end"' not in line or '"levels"' not in line:
        continue
    e = json.loads(line)
    rec = e.get("rec") or {}
    if e.get("status") != "ok" or not rec.get("levels") or (want and rec.get("id") not in want):
        continue
    lv = {round(x["N"], 2): x["t"][0] for x in rec["levels"]}
    if 0.5 not in lv or 3.0 not in lv:
        continue
    p0 = at(rec["t_rest0"][1])
    p_stop = at(rec["t_stop"])
    d = (p_stop - p0) / np.linalg.norm(p_stop - p0)
    a05, a3 = float((at(lv[0.5]) - p0) @ d), float((at(lv[3.0]) - p0) @ d)
    rows.append((rec["id"], a05, a3, a3 - a05, rec.get("fast", {}).get("contact")))
for i, a05, a3, rise, fc in rows:
    print(f"{i:24s} rest->0.5 N {a05:6.2f} mm, rest->3 N {a3:6.2f} mm, 0.5->3 N {rise:5.2f} mm, fast contact {fc}")
if len(rows) > 2:
    x = np.array([r_[3] for r_ in rows])
    print(f"0.5->3 N over {len(rows)} presses: median {np.median(x):.2f} mm, min {x.min():.2f}, max {x.max():.2f}")
