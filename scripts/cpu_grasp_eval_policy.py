#!/usr/bin/env python3
"""Score a trained residual policy (or the plan alone) on the exact E1 trial set.

Same cells as scripts/cpu_grasp_e1_openloop.py (same seed -> same placements and frictions), so the
numbers compare one-to-one with the open-loop baseline. Also records, per trial, the fraction of
grasp-phase control steps with any finger touching the table, and the mean residual size.

    .venv-cpu/bin/python scripts/cpu_grasp_eval_policy.py --policy runs/cpu_grasp/A_palm_s0_full/policy_best.pt \
        --out docs/experiments/20261102-cpu_grasp_policyA
    .venv-cpu/bin/python scripts/cpu_grasp_eval_policy.py --policy none --out ...   # plan alone, same metrics

Writes trials.csv, grid.csv, heatmap.png (and compare.png when --baseline points at an E1 grid.csv).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from morphohand.cpu_grasp import env as E  # noqa: E402

DIAMS = (20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0)
MASSES = (20, 40, 60, 80, 100, 125, 150)


def load_policy(path):
    """Returns (act_fn(obs)->actions, act_dim) for a cpu_grasp_train_ppo.py checkpoint."""
    from cpu_grasp_train_ppo import Agent, RunningNorm
    ck = torch.load(path, weights_only=False)
    sd = ck["agent"]
    obs_dim = sd["pi.0.weight"].shape[1]; act_dim = sd["pi.4.weight"].shape[0]
    priv_dim = sd["v.0.weight"].shape[1]
    agent = Agent(obs_dim, priv_dim, act_dim); agent.load_state_dict(sd); agent.eval()
    on = RunningNorm(obs_dim)
    on.mean, on.var, on.count = np.array(ck["onorm"]["mean"]), np.array(ck["onorm"]["var"]), ck["onorm"]["count"]

    def act(o):
        with torch.no_grad():
            return torch.tanh(agent.pi(torch.from_numpy(on(o)))).numpy()
    return act, act_dim, obs_dim


def e1_cells(trials, seed, mu_range):
    rng = np.random.default_rng(seed)
    cells = []
    for dm in DIAMS:
        for mg in MASSES:
            for _ in range(trials):
                pl = (*rng.uniform(-E.PLACE_XY, E.PLACE_XY, 2), rng.uniform(-E.PLACE_YAW, E.PLACE_YAW))
                cells.append((dm / 1000, mg / 1000, pl, float(rng.uniform(*mu_range))))
    return cells


def heat(ax, M, title):
    im = ax.imshow(M, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(MASSES)), MASSES); ax.set_yticks(range(len(DIAMS)), DIAMS)
    ax.set_xlabel("mass (g)"); ax.set_ylabel("diameter (mm)"); ax.set_title(title, fontsize=10)
    for i in range(len(DIAMS)):
        for j in range(len(MASSES)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] < 0.6 else "black")
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="checkpoint .pt, or 'none' for the plan alone")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default="docs/experiments/20261030-cpu_grasp_e1_v2/e1_grid.csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.policy == "none":
        act, act_dim, palm = None, 10, True
    else:
        act, act_dim, obs_dim = load_policy(a.policy)
        palm = act_dim == 10
    cfg = E.EnvCfg(n_envs=len(DIAMS) * len(MASSES) * a.trials, threads=12, seed=a.seed, palm_action=palm,
                   load_obs=(a.policy != "none" and obs_dim > 48))
    cells = e1_cells(a.trials, a.seed, cfg.mu_range)
    t0 = time.time()
    env = E.GraspEnv(cfg)
    o, _ = env.reset(cells=cells)
    n = cfg.n_envs
    infos = [None] * n
    floor_steps = np.zeros(n); alive_steps = np.zeros(n); res = np.zeros(n); dz_max = np.zeros(n)
    for _ in range(E.N_STEPS):
        alive = np.array([e.alive for e in env.envs])
        acts = np.zeros((n, env.act_dim), np.float32) if act is None else act(o)
        o, _, _, _, inf = env.step(acts)
        for i, e in enumerate(env.envs):
            if alive[i]:
                alive_steps[i] += 1
                floor_steps[i] += E._finger_floor(e.m, e.d, e.ix)
                res[i] += np.degrees(np.abs(e.dq)).mean()
                dz_max[i] = max(dz_max[i], abs(e.dz) * 1000)
        for i, x in enumerate(inf):
            if x:
                infos[i] = x
    env.pool.shutdown()
    print(f"{n} trials in {time.time() - t0:.0f} s")

    rows = []
    for i, (c, x) in enumerate(zip(cells, infos)):
        rows.append(dict(diameter_mm=c[0] * 1000, mass_g=round(c[1] * 1000), mu=c[3],
                         lifted=int(not x["dropped"]), held=int(x["held"]), success=int(x["success"]),
                         tilt_deg=x["tilt"], floor_frac=floor_steps[i] / max(1, alive_steps[i]),
                         floor_any=int(floor_steps[i] > 0), mean_abs_dq_deg=res[i] / max(1, alive_steps[i]),
                         max_abs_dz_mm=dz_max[i]))
    with open(os.path.join(a.out, "trials.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    S = np.zeros((len(DIAMS), len(MASSES))); L = np.zeros_like(S); F = np.zeros_like(S)
    with open(os.path.join(a.out, "grid.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["diameter_mm", "mass_g", "lifted", "success", "floor_any", "floor_frac"])
        for i, dm in enumerate(DIAMS):
            for j, mg in enumerate(MASSES):
                rs = [r for r in rows if r["diameter_mm"] == dm and r["mass_g"] == mg]
                L[i, j] = np.mean([r["lifted"] for r in rs]); S[i, j] = np.mean([r["success"] for r in rs])
                F[i, j] = np.mean([r["floor_any"] for r in rs])
                w.writerow([dm, mg, L[i, j], S[i, j], F[i, j], np.mean([r["floor_frac"] for r in rs])])
    ff = np.mean([r["floor_frac"] for r in rows])
    print(f"lifted {L.mean():.3f}  success {S.mean():.3f}  trials with any finger-table contact {F.mean():.3f}  "
          f"mean fraction of steps {ff:.3f}  mean |dq| {np.mean([r['mean_abs_dq_deg'] for r in rows]):.2f} deg  "
          f"max |dz| {np.mean([r['max_abs_dz_mm'] for r in rows]):.2f} mm (mean over trials)")
    for i, dm in enumerate(DIAMS):
        print(f"  {dm:4.1f} mm  " + "  ".join(f"{s:.2f}" for s in S[i]) + "   floor " + " ".join(f"{v:.2f}" for v in F[i]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    name = "plan alone" if a.policy == "none" else os.path.relpath(a.policy)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    heat(axes[0], L, f"lifted ({L.mean():.2f})")
    heat(axes[1], S, f"success ({S.mean():.2f})")
    im = heat(axes[2], F, f"trials with a finger touching the table ({F.mean():.2f})")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(f"{name}: E1 trial set, {a.trials} trials/cell, friction {cfg.mu_range}", fontsize=10)
    fig.savefig(os.path.join(a.out, "heatmap.png"), dpi=110, bbox_inches="tight")

    if a.baseline and os.path.exists(a.baseline):
        B = np.zeros_like(S)
        for r in csv.DictReader(open(a.baseline)):
            B[DIAMS.index(float(r["diameter_mm"])), MASSES.index(int(float(r["mass_g"])))] = float(r["success"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        heat(axes[0], B, f"open-loop plan: success {B.mean():.2f}")
        im = heat(axes[1], S, f"plan + residual policy: success {S.mean():.2f}")
        fig.colorbar(im, ax=axes, shrink=0.8)
        fig.suptitle("E1 trial set (same placements and frictions), held through the bench turn with tilt >= 45 deg",
                     fontsize=10)
        fig.savefig(os.path.join(a.out, "compare.png"), dpi=110, bbox_inches="tight")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
