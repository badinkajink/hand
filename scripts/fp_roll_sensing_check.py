#!/usr/bin/env python3
"""The roll on Sobol layouts that track poorly with exact sensing, on the CPU runner (2026-10-29, the foundational-pose
review page, Section "Sensing errors across the layouts").

On the GPU (rows/hom6d_gpu_sense_roll.jsonl), layouts whose roll held in every seed with an exact-sensing velocity RMSE
above 2 per step tracked at about 1.3 per step under the calibration-error variants. This runs twelve of them, evenly
spaced in the sorted list, through hom_tasks.rollout in seed 0 under four variants of hom_tasks.VARIANTS:
  rate50     exact sensing at 50 Hz;
  notip0     no fingertips, the exact object pose (contact frames from the witness points only);
  notip      no fingertips, the tag tracker's noise (0.5 mm, 1 deg) and 40 ms delay;
  cal_notip  as notip, with calibration offsets of sigma 1 deg per joint and 1 mm in the object pose.

  NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python scripts/fp_roll_sensing_check.py
Rows: docs/experiments/20261029-foundational_pose_review/rows/cpu_check_roll_poor.jsonl (one fsynced line per run).
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v1")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import fp_review_figures as FF  # noqa: E402
import hom_gpu as HG  # noqa: E402

VARIANTS = ("rate50", "notip0", "notip", "cal_notip")


def main():
    V = {v: {x["key"]: x for x in FF.load_hom("hom6d_gpu_sense_roll", v)} for v in ("rate50", "cal_notip")}
    poor = sorted(k for k in V["rate50"] if V["rate50"][k]["held"] >= 1 and V["cal_notip"][k]["held"] >= 1
                  and V["rate50"][k]["rmse"] > 2.0)
    pick = poor[::max(1, len(poor) // 12)][:12]
    scene = HG.Scene("sphere", 40.0)
    lays = {tuple(k["d"]): (k, mxy, setup) for k, mxy, setup in HG.layout_list("roll", "sobol8k", 0, 40.0)}
    path = HG.ROWS / "cpu_check_roll_poor.jsonl"
    if path.exists():
        path.unlink()
    print(f"{len(poor)} layouts with an exact-sensing RMSE above 2 per step; running {len(pick)}", flush=True)
    for d in pick:
        key, mxy, setup = lays[d]
        _bm, q_open = scene.prepare(mxy, setup)
        for var in VARIANTS:
            t0 = time.time()
            res = HG.cpu_rollout(scene, "roll", mxy, setup, q_open, 0, var)
            row = dict(key, task="roll", d_mm=40.0, seed=0, variant=var, status="ok", engine="mujoco_cpu",
                       gpu_rate50_rmse=V["rate50"][d]["rmse"], gpu_cal_notip_rmse=V["cal_notip"][d]["rmse"], **res)
            HG.T._append(path, row)
            print(var, res.get("held"), res.get("w_rmse_ratio"), f"{time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
