#!/usr/bin/env python3
"""The CPU runner on the grid's top layouts of a task (by the GPU's best-of-16 score, seeds 0-2), each from its best grasp,
seed 0, with all contacts and with the GPU evaluator's collision filter (pads and rigid spheres under 2 mm meet the object
only), and the periods in which those parts of two fingers touch on the CPU (2026-10-30: the twist peak's best grasp
held on the GPU and lost the index on the CPU, its index and middle caps touching).

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python scripts/hom_top_cpu_check.py twist 40 --n 20 [--part 0 --parts 2]
Rows: docs/experiments/20261029-foundational_pose_review/rows/cpu_best/top_<task>_d<d>.jsonl (fsynced, one per run).
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_films as F  # noqa: E402
import hom_landscape as L  # noqa: E402

HG, T, S, H = F.HG, F.T, F.S, F.H
OUT = F.ROWS / "cpu_best"
FINE = re.compile(r"_(dip_link|cassette|cap)_s\d+$")


def run(task, d, call, grasp, seed, gpu_filter):
    import mujoco
    _k, mxy, setup = F.find(task, "grid2d_size", d, call, grasp)
    scene = HG.Scene(T.SHAPE[task], d)
    bm, q_open = scene.prepare(mxy, setup)
    plant = T.Plant(scene.pads_path, setup)
    m = plant.m
    filt = np.array([("_pad" in (m.geom(g).name or "")) or bool(FINE.search(m.geom(g).name or "")
                                                                 and m.geom_size[g][0] < 0.002) for g in range(m.ngeom)])
    if gpu_filter:
        tool = np.flatnonzero(m.geom_bodyid == plant.tb)
        m.geom_contype[filt], m.geom_conaffinity[filt] = 2, 0
        m.geom_conaffinity[tool] = m.geom_conaffinity[tool] | 2
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for mm in (plant.m, m_pt):
        for f in F.FINGERS:
            mm.body_pos[mm.body(f"{f}_mount").id] = bm[f]
    S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = H.Mirror(m_pt, fs, os_)
    finger = {b: next((f for f in F.FINGERS if m.body(b).name.startswith(f + "_")), None) for b in range(m.nbody)}
    touch = Counter()
    adv = plant.advance

    def advance(t1):
        adv(t1)
        dd = plant.d
        pairs = set()
        for i in range(dd.ncon):
            c = dd.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            f1, f2 = finger[m.geom_bodyid[g1]], finger[m.geom_bodyid[g2]]
            if f1 and f2 and f1 != f2 and (filt[g1] or filt[g2]):
                pairs.add(tuple(sorted((f1, f2))))
        for p_ in pairs:
            touch[p_] += 1
    plant.advance = advance
    res, _tr, _ = T.rollout(task, setup, plant, mirror, q_open, seed, T.PRM, replace(H.HomParams(dt=T.DT_C)))
    return res, {f"{a}-{b}": n for (a, b), n in touch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("d", type=float)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    a = ap.parse_args()
    tab = [r for r in L.grid_table(a.task, a.d) if r.get("n_grasps")]
    top = sorted(tab, key=lambda r: -r["best"])[:a.n][a.part::a.parts]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"top_{a.task}_d{a.d:g}.jsonl"
    for r in top:
        for gf in (False, True):
            t0 = time.time()
            res, touch = run(a.task, a.d, r["call"], r["best_grasp"], 0, gf)
            row = {"task": a.task, "d_mm": a.d, "call": r["call"], "x_sep": r["x_sep"], "y_sep": r["y_sep"],
                   "grasp": r["best_grasp"], "gpu_best": r["best"], "seed": 0, "gpu_filter": gf,
                   "finger_touch_periods": touch, "score": L.rollout_score(a.task, res),
                   **{k: res.get(k) for k in ("held", "ramp_three_pads_max_deg", "w_rmse_ratio", "theta_rmse_deg",
                                              "link_force_max_N")}}
            T._append(path, row)
            print(r["call"], (r["x_sep"], r["y_sep"]), "filter" if gf else "full", f"score {row['score']:.2f} (GPU "
                  f"{r['best']:.2f})", touch, f"{time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
