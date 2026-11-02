#!/usr/bin/env python3
"""Turn raw single-pendulum recordings into BAM (Better Actuator Models) fitting logs.

    ../.venv-bam/bin/python prepare_logs.py [--raw data_raw/single] [--out data_processed/single]

Per log: keeps status == "ok" only; resamples to a fixed dt (as bam.process does);
adds top-level first_moment [kg m] and inertia [kg m^2] for bam_scs0009.RealPendulum,
with the cylinder's own spin inertia recomputed as a solid cylinder (the recorder used
the 30 mm washer formula); drops the nested real_pendulum dict (bam.logs.make_batch
vectorizes every top-level key).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from bam_rig_params import ARMS

CYL_D_MM = 19.88  # steel cylinder 19.88 mm dia x 19.88 mm long (measured), axis parallel to the servo shaft


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data_raw/single")
    ap.add_argument("--out", default="data_processed/single")
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--cyl-d-mm", type=float, default=CYL_D_MM)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(f"{a.raw}/*.json")):
        d = json.load(open(f))
        if d.get("status") != "ok":
            continue
        rp = d.pop("real_pendulum")
        m0, com, izz, _ = ARMS[rp["arm"]]
        k = rp["arm_g"] / m0
        m, L = rp["stack_g"], rp["hole_mm"]
        i_cyl = m * (a.cyl_d_mm / 2) ** 2 / 2
        d["first_moment"] = (m * L + k * m0 * com) * 1e-6
        d["inertia"] = (m * L**2 + k * izz + i_cyl) * 1e-9
        d["hole_mm"] = L

        E = d["entries"]
        t = np.array([e["timestamp"] for e in E])
        t = t - t[0]
        ts = np.arange(0.0, t[-1], a.dt)
        out = []
        cols = {key: np.array([float(e[key]) for e in E]) for key in E[0] if key != "timestamp"}
        res = {key: np.interp(ts, t, v) for key, v in cols.items()}
        for i, tt in enumerate(ts):
            e = {key: float(res[key][i]) for key in res}
            e["torque_enable"] = bool(res["torque_enable"][i] > 0.5)
            e["timestamp"] = float(tt)
            out.append(e)
        d["entries"] = out
        d["dt"] = a.dt
        json.dump(d, open(os.path.join(a.out, os.path.basename(f)), "w"))
        n += 1
    print(f"{n} logs -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
