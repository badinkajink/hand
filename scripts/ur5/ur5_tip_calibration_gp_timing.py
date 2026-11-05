#!/usr/bin/env python3
"""Inference time of the contact viewer's board GP (wrench_viewer.BoardGP), one board reading at a time on one core:
the full prediction (mean over every training hold, then the posterior variance from the 256 holds of largest kernel
weight), the mean alone, and three boards a frame. Readings are the rig holds of v3 ID 3 (11-04).

    OPENBLAS_NUM_THREADS=1 python3 scripts/ur5/ur5_tip_calibration_gp_timing.py [MODEL.npz]
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/sr2hand/driver/reskin5x/host"))
import wrench_viewer as WV  # noqa: E402

DATA = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data"
path = sys.argv[1] if len(sys.argv) > 1 else str(DATA / "20261105-board_gp_v3_id1_id3_rig_id3_id6.npz")
g = WV.BoardGP(path)
D = np.load(ROOT / "docs/experiments/20261030-inverted_rig_seat_block/data/holds/20261104-145215-id3pts11/holds.npz",
            allow_pickle=True)["d_board"].astype(float)
D = D[np.isfinite(D).all(1)][:300]
for d in D[:20]:
    g.predict(d)


def per_call(fn, reps=3):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        for d in D:
            fn(d)
        best = min(best, (time.perf_counter() - t) / len(D))
    return best * 1e3


def mean_only(d):
    q = (d - g.mx) / g.sx
    k = np.exp(-0.5 * ((g.Z - q) ** 2).sum(1) / g.ell ** 2)
    return k @ g.a * g.sy + g.my


full, mean = per_call(g.predict), per_call(mean_only)
print(f"{path.split('/')[-1]}: {g.n} training holds, m {g.m}")
print(f"full prediction {full:.2f} ms a reading, mean alone {mean:.2f} ms; three boards a frame {3 * full:.2f} ms "
      f"(10 ms at 100 Hz)")
