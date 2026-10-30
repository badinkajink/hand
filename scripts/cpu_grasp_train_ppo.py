#!/usr/bin/env python3
"""Compact PPO (proximal policy optimization) for the CPU grasp env: residual policy on the plan.

Actor: obs (48 with the palm action, +45 load values in condition C) -> 10 residual actions in
[-1, 1] (9 finger targets + the UR5 palm height; --no-palm for the 9-action version) (tanh-squashed
Gaussian mean, state-independent log std). Critic: privileged obs (actor obs + object state,
contacts, size, mass, friction). Every iteration resets all envs and runs one full grasp episode
(120 control steps + the judge), so there is no bootstrapping across episodes.

    .venv-cpu/bin/python scripts/cpu_grasp_train_ppo.py --cond A --iters 150 --out runs/cpu_grasp/A_s0

Every --eval-every iterations: deterministic evaluation on the E1 grid (4 trials/cell, fixed seed),
logged next to the open-loop baseline. Checkpoints: <out>/policy_<iter>.pt and policy_best.pt.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from morphohand.cpu_grasp import env as E  # noqa: E402

DIAMS = (20.0, 22.5, 25.0, 27.5, 30.0, 32.5, 35.0, 37.5, 40.0)
MASSES = (20, 40, 60, 80, 100, 125, 150)


class RunningNorm:
    def __init__(self, n):
        self.mean, self.var, self.count = np.zeros(n), np.ones(n), 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), len(x)
        d = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(np.float32)

    def state(self):
        return dict(mean=self.mean.tolist(), var=self.var.tolist(), count=float(self.count))


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, o))


class Agent(nn.Module):
    def __init__(self, obs_dim, priv_dim, act_dim=9):
        super().__init__()
        self.pi = mlp(obs_dim, act_dim)
        self.v = mlp(priv_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.7))
        nn.init.zeros_(self.pi[-1].weight); nn.init.zeros_(self.pi[-1].bias)   # start = the plan

    def dist(self, o):
        mu = torch.tanh(self.pi(o))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))


def evaluate(agent, onorm, cond, palm=True, trials=4, seed=123):
    """Deterministic policy on the E1 grid; returns (success, lifted, per-cell success array)."""
    rng = np.random.default_rng(seed)
    cfg = E.EnvCfg(n_envs=len(DIAMS) * len(MASSES) * trials, threads=12, load_obs=(cond == "C"), seed=seed,
                   palm_action=palm)
    cells = [(dm / 1000, mg / 1000,
              (*rng.uniform(-E.PLACE_XY, E.PLACE_XY, 2), rng.uniform(-E.PLACE_YAW, E.PLACE_YAW)),
              float(rng.uniform(*cfg.mu_range)))
             for dm in DIAMS for mg in MASSES for _ in range(trials)]
    env = E.GraspEnv(cfg)
    o, _ = env.reset(cells=cells)
    infos = [None] * cfg.n_envs
    for _ in range(E.N_STEPS):
        with torch.no_grad():
            a = torch.tanh(agent.pi(torch.from_numpy(onorm(o)))).numpy()
        o, _, _, _, inf = env.step(a)
        for i, x in enumerate(inf):
            if x:
                infos[i] = x
    env.pool.shutdown()
    s = np.array([x["success"] for x in infos], float).reshape(len(DIAMS), len(MASSES), trials).mean(2)
    lifted = 1 - np.mean([x["dropped"] for x in infos])
    return float(s.mean()), float(lifted), s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="A", choices=["A", "C"])
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--out", default="runs/cpu_grasp/A_s0")
    ap.add_argument("--no-palm", action="store_true", help="9 finger actions only (no UR5 z residual)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.set_num_threads(2)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)

    palm = not a.no_palm
    cfg = E.EnvCfg(n_envs=a.envs, threads=12, load_obs=(a.cond == "C"), seed=a.seed, palm_action=palm)
    env = E.GraspEnv(cfg)
    K = env.act_dim
    agent = Agent(env.obs_dim, env.priv_dim, K)
    opt = torch.optim.Adam(agent.parameters(), lr=a.lr)
    onorm, pnorm = RunningNorm(env.obs_dim), RunningNorm(env.priv_dim)
    gamma, lam, clip = 0.99, 0.95, 0.2
    T, N = E.N_STEPS, a.envs
    log = open(os.path.join(a.out, "train_log.csv"), "w", newline="")
    lw = csv.writer(log)
    lw.writerow(["iter", "time_s", "ep_return", "lifted", "success", "tilt", "eval_success", "eval_lifted",
                 "std", "v_loss", "pi_loss", "kl"])
    best, t0 = -1.0, time.time()
    for it in range(1, a.iters + 1):
        o, p = env.reset()
        onorm.update(o); pnorm.update(p)
        O = np.zeros((T, N, env.obs_dim), np.float32); P = np.zeros((T, N, env.priv_dim), np.float32)
        A = np.zeros((T, N, K), np.float32); LP = np.zeros((T, N), np.float32)
        R = np.zeros((T, N), np.float32); V = np.zeros((T, N), np.float32); M = np.zeros((T, N), bool)
        alive = np.ones(N, bool)
        infos = [None] * N
        for t in range(T):
            on, pn = onorm(o), pnorm(p)
            with torch.no_grad():
                dist = agent.dist(torch.from_numpy(on))
                act = dist.sample()
                lp = dist.log_prob(act).sum(-1)
                v = agent.v(torch.from_numpy(pn)).squeeze(-1)
            O[t], P[t], A[t], LP[t], V[t], M[t] = on, pn, act.numpy(), lp.numpy(), v.numpy(), alive
            o, p, r, d, inf = env.step(np.clip(act.numpy(), -1, 1))
            R[t] = r * alive
            for i, x in enumerate(inf):
                if x:
                    infos[i] = x
            alive = alive & ~d
            if t % 20 == 0:
                onorm.update(o[alive] if alive.any() else o); pnorm.update(p[alive] if alive.any() else p)
        # GAE; an env's trajectory ends at its last alive step (no bootstrap)
        adv = np.zeros((T, N), np.float32); last = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nv = V[t + 1] * M[t + 1] if t + 1 < T else np.zeros(N, np.float32)
            nm = M[t + 1] if t + 1 < T else np.zeros(N, bool)
            delta = R[t] + gamma * nv - V[t]
            last = delta + gamma * lam * nm * last
            adv[t] = last * M[t]
        ret = adv + V
        idx = np.flatnonzero(M.reshape(-1))
        fo = torch.from_numpy(O.reshape(-1, env.obs_dim)[idx]); fp = torch.from_numpy(P.reshape(-1, env.priv_dim)[idx])
        fa = torch.from_numpy(A.reshape(-1, K)[idx]); flp = torch.from_numpy(LP.reshape(-1)[idx])
        fadv = adv.reshape(-1)[idx]; fadv = torch.from_numpy((fadv - fadv.mean()) / (fadv.std() + 1e-8))
        fret = torch.from_numpy(ret.reshape(-1)[idx])
        n = len(idx)
        for _ in range(a.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, a.minibatch):
                b = perm[s:s + a.minibatch]
                dist = agent.dist(fo[b])
                lp = dist.log_prob(fa[b]).sum(-1)
                ratio = (lp - flp[b]).exp()
                pl = -torch.min(ratio * fadv[b], ratio.clamp(1 - clip, 1 + clip) * fadv[b]).mean()
                vl = ((agent.v(fp[b]).squeeze(-1) - fret[b]) ** 2).mean()
                loss = pl + 0.5 * vl - 0.001 * dist.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()
            with torch.no_grad():
                kl = float((flp - agent.dist(fo).log_prob(fa).sum(-1)).mean())
            if kl > 0.03:
                break
        succ = np.mean([x["success"] for x in infos]); lifted = 1 - np.mean([x["dropped"] for x in infos])
        tilt = np.mean([x["tilt"] for x in infos])
        es, el = "", ""
        if it % a.eval_every == 0 or it == a.iters:
            es, el, grid = evaluate(agent, onorm, a.cond, palm)
            torch.save(dict(agent=agent.state_dict(), onorm=onorm.state(), it=it, eval_success=es),
                       os.path.join(a.out, f"policy_{it:04d}.pt"))
            if es > best:
                best = es
                torch.save(dict(agent=agent.state_dict(), onorm=onorm.state(), it=it, eval_success=es),
                           os.path.join(a.out, "policy_best.pt"))
                np.save(os.path.join(a.out, "best_eval_grid.npy"), grid)
        lw.writerow([it, round(time.time() - t0), R.sum(0).mean(), lifted, succ, tilt, es, el,
                     float(agent.log_std.exp().mean().detach()), float(vl.detach()), float(pl.detach()), kl]); log.flush()
        print(f"it {it:4d} {time.time() - t0:6.0f}s  return {R.sum(0).mean():6.2f}  lifted {lifted:.2f}  "
              f"success {succ:.2f}  tilt {tilt:5.1f}" + (f"  | EVAL success {es:.3f} lifted {el:.2f}" if es != "" else ""),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
