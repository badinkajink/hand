#!/usr/bin/env python3
"""E1: open-loop baseline on the CPU grasp env (manta-hand final paper, experiment 1).

The plan (plan.py: table grasp, UR5 palm z, 2 mm fingertip clearance, PIP >= 0, 8 mm squeeze) runs
with zero residual on a diameter x mass grid; every trial draws its own placement error
(+-2 mm, +-2 deg) and pad-object friction (EnvCfg.mu_range); the judge (judge.py, D1 bench turn)
scores what the plan delivers.

    .venv-cpu/bin/python scripts/cpu_grasp_e1_openloop.py [--trials 16] [--out docs/experiments/<dir>]

Writes e1_trials.csv (one row per trial), e1_grid.csv (per cell) and e1_heatmap.png.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from morphohand.cpu_grasp import env as E  # noqa: E402

DIAMS = (20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0)    # mm
MASSES = (20, 40, 60, 80, 100, 125, 150)                         # g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--out", default="docs/experiments/20261030-cpu_grasp_e1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    cfg = E.EnvCfg(n_envs=len(DIAMS) * len(MASSES) * a.trials, threads=12, seed=a.seed)
    cells = []
    for dm in DIAMS:
        for mg in MASSES:
            for _ in range(a.trials):
                pl = (*rng.uniform(-E.PLACE_XY, E.PLACE_XY, 2), rng.uniform(-E.PLACE_YAW, E.PLACE_YAW))
                cells.append((dm / 1000, mg / 1000, pl, float(rng.uniform(*cfg.mu_range))))
    t0 = time.time()
    env = E.GraspEnv(cfg)
    env.reset(cells=cells)
    n = cfg.n_envs
    infos = [None] * n
    for _ in range(E.N_STEPS):
        _, _, _, _, inf = env.step(np.zeros((n, env.act_dim), np.float32))
        for i, x in enumerate(inf):
            if x:
                infos[i] = x
    print(f"{n} trials in {time.time() - t0:.0f} s")

    rows = []
    for c, x, e in zip(cells, infos, env.envs):
        v = getattr(e, "verdict", None)
        rows.append(dict(diameter_mm=c[0] * 1000, mass_g=round(c[1] * 1000), dx_mm=c[2][0] * 1000,
                         dy_mm=c[2][1] * 1000, dyaw_deg=np.degrees(c[2][2]), mu=c[3],
                         lifted=int(not x["dropped"]), held=int(x["held"]), success=int(x["success"]),
                         tilt_deg=x["tilt"], max_tilt_deg=(v.max_tilt_deg if v and not x["dropped"] else 0.0)))
    with open(os.path.join(a.out, "e1_trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    S = np.zeros((len(DIAMS), len(MASSES))); L = np.zeros_like(S)
    with open(os.path.join(a.out, "e1_grid.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["diameter_mm", "mass_g", "lifted", "success", "mean_tilt_deg"])
        for i, dm in enumerate(DIAMS):
            for j, mg in enumerate(MASSES):
                rs = [r for r in rows if r["diameter_mm"] == dm and r["mass_g"] == mg]
                L[i, j] = np.mean([r["lifted"] for r in rs]); S[i, j] = np.mean([r["success"] for r in rs])
                w.writerow([dm, mg, L[i, j], S[i, j], np.mean([r["tilt_deg"] for r in rs])])
    print(f"lifted {L.mean():.2f}   success {S.mean():.2f}")
    print("success by diameter (rows) x mass (cols g):", MASSES)
    for i, dm in enumerate(DIAMS):
        print(f"  {dm:4.1f} mm  " + "  ".join(f"{s:.2f}" for s in S[i]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, M, title in ((axes[0], L, "lifted"), (axes[1], S, "success (held through the bench turn, tilt >= 45 deg)")):
        im = ax.imshow(M, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(MASSES)), MASSES); ax.set_yticks(range(len(DIAMS)), DIAMS)
        ax.set_xlabel("mass (g)"); ax.set_ylabel("diameter (mm)"); ax.set_title(title, fontsize=10)
        for i in range(len(DIAMS)):
            for j in range(len(MASSES)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < 0.6 else "black")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(f"E1 open loop, {a.trials} trials/cell, friction {cfg.mu_range}, placement +-2 mm / +-2 deg", fontsize=10)
    fig.savefig(os.path.join(a.out, "e1_heatmap.png"), dpi=110, bbox_inches="tight")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
