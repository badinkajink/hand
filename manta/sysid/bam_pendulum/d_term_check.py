#!/usr/bin/env python3
"""How does the SCS0009 firmware's D gain enter its PWM output? Uses the load register
(= duty x 1000) in double-pendulum logs at D = 0 ... 150.  ../.venv-bam/bin/python d_term_check.py"""
import glob
import json
import collections

import numpy as np

KP_G, MIN_PWM, DZ = 0.189, 0.036, np.radians(300 / 1024)
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob("data_raw/double_v2*/*.json")):
    d = json.load(open(f))
    if d.get("status") != "ok" or "raw_sign" not in d:
        continue
    kp, kd = d["kp"], d.get("kd", 0)
    E = [e for e in d["entries"] if e["phase"] == "path"]
    t = np.array([e["timestamp"] for e in E])
    for j in range(2):
        q = np.radians([e["position"][j] for e in E]); g = np.radians([e["goal_position"][j] for e in E])
        y = -np.array([e["load"][j] for e in E], float) / 1000
        # velocity over +-20 ms windows (position is quantized to 0.29 deg)
        tt = np.arange(t[0], t[-1], 0.004)
        qq, gg = np.interp(tt, t, q), np.interp(tt, t, g)
        k = 5
        v = np.r_[np.zeros(k), (qq[2 * k:] - qq[:-2 * k]) / (tt[2 * k:] - tt[:-2 * k]), np.zeros(k)]
        gv = np.r_[np.zeros(k), (gg[2 * k:] - gg[:-2 * k]) / (tt[2 * k:] - tt[:-2 * k]), np.zeros(k)]
        yy = np.interp(tt, t, y)
        e = gg - qq
        p_part = KP_G * kp * e + MIN_PWM * np.sign(e) * (np.abs(e) > DZ)
        m = (np.abs(yy) < 0.95) & (np.abs(p_part) < 0.95)
        r = rows[(kp, kd)]
        for key, val in (("res", yy - p_part), ("v", v), ("ev", gv - v), ("y", yy), ("p", p_part)):
            r[key] += list(val[m])
print(f"{'P':>3} {'D':>4} | resid rms (P-law only) | fit resid = -b*kd*v : b, rms | fit b*kd*(d err/dt): b, rms")
B = []
for (kp, kd), r in sorted(rows.items()):
    res, v, ev = (np.array(r[k]) for k in ("res", "v", "ev"))
    out = f"{kp:3d} {kd:4d} | {np.sqrt(np.mean(res**2)):.4f}"
    if kd:
        for x in (-kd * v, kd * ev):
            b = float(np.dot(x, res) / np.dot(x, x))
            out += f" | b {b:.5f} rms {np.sqrt(np.mean((res - b * x) ** 2)):.4f}"
        B.append((kp, kd, b))
    print(out)
