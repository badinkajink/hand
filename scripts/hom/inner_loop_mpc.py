#!/usr/bin/env python3
"""Sampling MPC on the SR2 hand in MuJoCo-Warp: the training-free baselines of the inner-loop benchmark (owner,
2026-11-12: "let's try MPC. In addition to the typical cem, mppi, predictive sampling, we should see if dial-MPC
(annealed sampling) which we implemented on CPU finally yields better results now that we have warp, and maybe the
worst case sampling from spider (which uses annealed sampling) can be broadly used").

Plant and task are the benchmark's (uhas_sr2.UBatch, the twist of the 40 mm sphere). HOM drives the approach, the
closing, the squeeze and the lift. From the task phase (t = 2.4 s) a planner sets the nine servo targets. The planner's
model is the plant's: the same scene, pads, servo and 1 ms physics in MuJoCo-Warp. Every REPLAN periods the state of
each plant world (positions, velocities, servo targets, the post and the solver's warm start) is copied into the
planner's worlds of that plant world. A plan is the nine targets at KNOTS knots over H periods, linear between the
knots, with the first knot pinned to the target applied now. The plant applies the plan's first REPLAN periods.

Cost of a rollout, summed over its periods: the task's terms of J, then the limit terms that the UHAS rerun with
limits also has in its reward (uhas_sr2.REW_LIMITS):
    (rotation error / 5 deg)^2 + (distance from the start position / 3 mm)^2 + 2 per pad under 0.05 N
    + 100 per period after the sphere fell 15 mm
    + 0.5 sum_j clip((servo demand_j / torque limit - 0.8) / 0.2, 0, 3), the demand kp |target - q|
    + 0.25 sum_j clip((3 deg - distance to the joint's range limit) / 3 deg, 0, 1)
and, per gap between knots, sum_j (rate_j / 20 deg/s)^2.

Methods. Each spends iters x n rollouts per replanning (n = --n samples per plant world and iteration; spider_wc n / 3
samples in each of three friction groups); the update rules are those of hand_object_scale_mpc.Planner (2026-10-13)
and of SPIDER:
    ps        predictive sampling: the cheapest of the nominal and its noisy copies (Howell et al. 2022)
    mppi      an MPPI step per iteration: weights exp(-(c - c_min) / lam), lam = 0.03 std(c)
    cem       cross-entropy: the 15 % cheapest set the mean and std (momentum 0.1, floor 0.005 rad); the mean executed
    icem      CEM with AR(1) knot noise (0.7), the two best kept into the next replanning; the best sample executed
    anneal    DIAL-MPC's annealed sampling as in mujoco_mpc (the CPU planner of 2026-10-13): knot noise
              sigma exp(-i / (0.5 iters) - (K - 1 - k) / (0.5 K)) in iteration i at knot k, an MPPI step after each
    spider    SPIDER's optimizer (Pan et al. 2025; spider/optimizers/sampling.py): knot noise from 0.5 to 1.0 sigma
              along the horizon, times 0.1^(i / iters) in iteration i; the first sample noise-free, the last 1 % at
              0.01; a softmax at temperature 0.3 over the best 10 % of the normalized rewards (-cost per period); stop
              when no sample beats the nominal by 0.01
    spider_wc the same, each sample's reward the worst over the scene frictions 0.4, 0.7 and 1.0 (SPIDER's domain
              randomization groups), n / 3 samples per iteration
The script also runs HOM through the task (method hom) and a trained UHAS policy (method uhas, --run), so that every
controller is scored on the same plant, pairs and seeds.

    .venv/bin/python scripts/hom/inner_loop_mpc.py pairs
    .venv/bin/python scripts/hom/inner_loop_mpc.py timing --chunk 16 --n 128
    .venv/bin/python scripts/hom/inner_loop_mpc.py run --method anneal [--seeds 0 1] [--plant-mu 1.0] [--chunk 16]
Pairs: rows/mpc_pairs_twist.json, the ten test layouts with HOM's best J and the ten with UHAS's best J, each with that
controller's best grasp. Rows: docs/experiments/20261112-inner_loop_benchmark/rows/mpc_<method>_twist[_mu<mu>].jsonl,
one per world, appended and fsynced per chunk; a rerun skips the worlds already written.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import hom_gpu as HG  # noqa: E402
import inner_loop_usability as IU  # noqa: E402
import uhas_sr2 as UH  # noqa: E402
import uhas_sr2_cik as U  # noqa: E402
import warp as wp  # noqa: E402
from sr2hand.rl.math import quat_conj, quat_from_axis_angle, quat_mul  # noqa: E402

T = HG.T
TASK = "twist"
ROWS = UH.OUT / "rows"
RUN = "20261112-1244_twist_s0"           # the UHAS run whose test split and usability rows choose the pairs
REPLAN, H, KNOTS = 10, 30, 3             # periods between replannings, horizon (periods), knots (the first pinned)
COST = dict(rot_deg=5.0, pos_mm=3.0, pad=2.0, f_on=0.05, drop=100.0, drop_m=0.015, tau=0.5, tau_on=0.8, q=0.25,
            q_band_deg=3.0, rate=1.0, rate_dps=20.0)
PLAN = dict(iters=4, sigma=0.08, lam_rel=0.03, elite_frac=0.15, momentum=0.1, sigma_min=0.005,
            ar_alpha=0.7, keep=2, beta1=0.5, beta2=0.5, sp_first=0.5, sp_last=1.0, sp_final=0.1, sp_exploit=0.01,
            sp_exploit_scale=0.01, sp_temp=0.3, sp_top=0.1, sp_improve=0.01, frictions=(0.4, 0.7, 1.0))
METHODS = ("ps", "mppi", "cem", "icem", "anneal", "spider", "spider_wc")
KNOT_T = np.linspace(0.0, H, KNOTS)


def interp(times, knot_t=KNOT_T):
    """(len(times), K): the linear weights of the knots at each time, in periods (the last knot held beyond it)."""
    K = len(knot_t)
    w = np.stack([np.interp(times, knot_t, np.eye(K)[k]) for k in range(K)], 1)
    return torch.tensor(w, dtype=torch.float32, device="cuda")


W_ROLL = interp(np.arange(1, H + 1))          # the target of rollout period h (it ends at h + 1)
W_EXEC = interp(np.arange(1, REPLAN + 1))     # the plant's targets until the next replanning
W_SHIFT = interp(REPLAN + KNOT_T)             # the knots REPLAN periods later


def configure(replan, horizon, knots, iters):
    """Another replanning interval, horizon, knot count or iteration count (periods); the interpolation tables
    again."""
    global REPLAN, H, KNOTS, KNOT_T, W_ROLL, W_EXEC, W_SHIFT
    REPLAN, H, KNOTS = int(replan), int(horizon), int(knots)
    PLAN["iters"] = int(iters)
    KNOT_T = np.linspace(0.0, H, KNOTS)
    W_ROLL = interp(np.arange(1, H + 1), KNOT_T)
    W_EXEC = interp(np.arange(1, REPLAN + 1), KNOT_T)
    W_SHIFT = interp(REPLAN + KNOT_T, KNOT_T)


# ------------------------------------------------------------------------------------------------ worlds
def pairs(run=RUN, n_top=10):
    """The (layout, grasp) pairs: the n_top test layouts with HOM's best J, each with its best grasp by the grid's mean
    over seeds 0-2, and the n_top with UHAS's best J, each with its best grasp by UHAS's J at seed 0 (the usability
    rows); a pair chosen twice is kept once."""
    import inner_loop_benchmark_page as BP
    sp = json.loads((UH.RUNS / run / "split.json").read_text())
    test = set(map(tuple, sp["test"]))
    hom = {}
    for (x, y, g), v in BP.j_rows(TASK, UH.D_MM[TASK]).items():
        if (x, y) in test:
            j = float(np.mean([a for a, _h in v]))
            if (x, y) not in hom or j > hom[(x, y)][1]:
                hom[(x, y)] = (g, j)
    uh = {}
    for line in (ROWS / f"usability_{TASK}_{run}.jsonl").read_text().splitlines():
        r = json.loads(line)
        k = (r["x_sep"], r["y_sep"])
        if r["controller"] == "uhas" and (k not in uh or r["J"] > uh[k][1]):
            uh[k] = (r["grasp"], r["J"])
    out, seen = [], set()
    for by, d in (("hom", hom), ("uhas", uh)):
        for k in sorted(d, key=lambda k_: -d[k_][1])[:n_top]:
            if (k, d[k][0]) not in seen:
                seen.add((k, d[k][0]))
                out.append({"x": k[0], "y": k[1], "grasp": int(d[k][0]), "by": by, "J_by": round(d[k][1], 4)})
    return out


def worlds_of(prs, seeds):
    scene = HG.Scene(T.SHAPE[TASK], UH.D_MM[TASK])
    want = {(p["x"], p["y"], p["grasp"]): p["by"] for p in prs}
    out = []
    for key, mxy, setup in UH.grid_entries(TASK):
        k = (*UH.xy_of(key), int(key.get("grasp", 0)))
        if k not in want:
            continue
        bm, q_open = scene.prepare(mxy, setup)
        for sd in seeds:
            out.append({"key": key, "seed": sd, "setup": setup, "body_mounts": bm, "q_open": q_open, "by": want[k],
                        "var": HG.variant_world(UH.VARIANT[TASK], sd)})
    return scene, out


def set_friction(B, mu):
    """Every geom's sliding friction set to mu[w] in world w (MuJoCo-Warp's per-world geom_friction), and the batch's
    graphs captured again: a captured graph keeps the arrays it was captured with."""
    fr = np.repeat(B.m.geom_friction[None].astype(np.float32), B.W, 0)
    fr[:, :, 0] = np.asarray(mu, np.float32)[:, None]
    B.wm.geom_friction = wp.array(fr, dtype=wp.vec3)
    B._capture()


# ------------------------------------------------------------------------------------------------ rollouts
class Rollouts:
    """The planner's worlds, n samples times G friction groups per plant world, in the plant's scene and physics;
    world (p G + g) n + k is sample k of plant world p in group g."""

    def __init__(self, scene, chunk, n, frictions=None):
        self.P, self.n = len(chunk), n
        self.mu = tuple(frictions) if frictions else None
        self.G = len(self.mu) if self.mu else 1
        self.B = UH.UBatch(scene, TASK, self.rep(chunk), T.DT_C, gen_seed=0)
        if self.mu:
            set_friction(self.B, np.tile(np.repeat(np.asarray(self.mu, np.float32), n), self.P))
        B = self.B
        self.q9 = torch.tensor(np.asarray(B.qadr).reshape(-1), dtype=torch.long, device="cuda")
        self.kp = torch.tensor(B.kp.reshape(-1), dtype=torch.float32, device="cuda")
        self.lim = IU.Limits(B)

    def rep(self, chunk):
        return [w for w in chunk for _ in range(self.G * self.n)]

    def begin(self, chunk):
        self.B.begin(self.rep(chunk), hom=False)

    def load(self, st):
        S = self.B.state()
        r = self.G * self.n
        for k in ("qpos", "qvel", "ctrl", "mocap_pos", "warm"):
            if k in S and k in st:
                S[k].copy_(st[k].repeat_interleave(r, 0))

    def rollout(self, Uh, ctx):
        """The cost of each world's targets Uh (W, H, 9) from the loaded state (W,): the task's terms and the limit
        terms summed over the periods."""
        B, vw, lim = self.B, self.B.vw, self.lim
        c = torch.zeros(B.W, device="cuda")
        drop = torch.zeros(B.W, dtype=torch.bool, device="cuda")
        band = math.radians(COST["q_band_deg"])
        lim_on = lim.limited.float()
        for h in range(H):
            vw["ctrl"][:, B.act_t] = Uh[:, h]
            wp.capture_launch(B.graph)
            qp = vw["qpos"]
            p = qp[:, B.qa:B.qa + 3] - ctx["palm"]
            dq = quat_mul(qp[:, B.qa + 3:B.qa + 7], quat_conj(ctx["goal"][:, h]))
            err = 2.0 * torch.asin(dq[:, 1:].norm(dim=-1).clamp(max=1.0))
            drop |= p[:, 2] < ctx["p0"][:, 2] - COST["drop_m"]
            q = qp[:, self.q9]
            dmd = self.kp * (Uh[:, h] - q).abs() / lim.tau
            near = torch.minimum(q - lim.lo, lim.hi - q)
            c += ((err / math.radians(COST["rot_deg"])) ** 2
                  + ((p - ctx["p0"]).norm(dim=-1) / (1e-3 * COST["pos_mm"])) ** 2
                  + COST["pad"] * (B.ft_t < COST["f_on"]).float().sum(-1) + COST["drop"] * drop.float()
                  + COST["tau"] * ((dmd - COST["tau_on"]) / (1.0 - COST["tau_on"])).clamp(0.0, 3.0).sum(-1)
                  + COST["q"] * (((band - near) / band).clamp(0.0, 1.0) * lim_on).sum(-1))
        return torch.nan_to_num(c, nan=1e6, posinf=1e6)


# ------------------------------------------------------------------------------------------------ planners
class MPC:
    """One planner for P plant worlds: the nominal knots (P, K, 9) and the update rule of its method."""

    def __init__(self, method, P, lo, hi, n, G, seed):
        self.m, self.P, self.n, self.G = method, P, n, G
        self.lo, self.hi = lo, hi
        self.gen = torch.Generator(device="cuda")
        self.gen.manual_seed(1000 + seed)
        self.nom, self.elites = None, None
        self.ar = torch.arange(P, device="cuda")

    def noise(self, n, scale):
        e = torch.randn((self.P, n, KNOTS, 9), generator=self.gen, device="cuda")
        if self.m == "icem":
            a = PLAN["ar_alpha"]
            for k in range(1, KNOTS):
                e[:, :, k] = a * e[:, :, k - 1] + math.sqrt(1.0 - a * a) * e[:, :, k]
        return e * scale

    def constrain(self, S, u):
        S = torch.maximum(torch.minimum(S, self.hi), self.lo)
        S[:, :, 0] = u[:, None]
        return S

    def rate_cost(self, S):
        gap = float(KNOT_T[1] - KNOT_T[0]) * T.DT_C
        r = torch.diff(S, dim=2) / gap / math.radians(COST["rate_dps"])
        return COST["rate"] * (r ** 2).sum((-1, -2))

    def evaluate(self, R, st, S, ctx):
        """Costs (P, n) of the samples S (P, n, K, 9), each the worst over the friction groups."""
        P, n = S.shape[:2]
        Uh = torch.einsum("hk,pnkj->pnhj", W_ROLL, S)
        Uh = Uh[:, None].expand(P, self.G, n, H, 9).reshape(P * self.G * n, H, 9)
        R.load(st)
        c = R.rollout(Uh, ctx).reshape(P, self.G, n).max(1).values
        return c + self.rate_cost(S)

    def pick(self, S, idx):
        return S[self.ar, idx]

    def plan(self, R, st, u, ctx):
        """The knots (P, K, 9) to execute from the plant state st; u (P, 9) the targets applied now."""
        if self.nom is None:
            self.nom = u[:, None].expand(self.P, KNOTS, 9).clone()
        else:
            self.nom = torch.einsum("ik,pkj->pij", W_SHIFT, self.nom)
            if self.elites is not None:
                self.elites = torch.einsum("ik,pekj->peij", W_SHIFT, self.elites)
        m, I, n, sig = self.m, PLAN["iters"], self.n, PLAN["sigma"]
        nom0 = self.constrain(self.nom[:, None], u)
        if m == "ps":
            best_c = torch.full((self.P,), float("inf"), device="cuda")
            best = nom0[:, 0].clone()
            for i in range(I):
                S = self.constrain(self.nom[:, None] + self.noise(n, sig), u)
                if i == 0:
                    S[:, 0] = nom0[:, 0]
                c = self.evaluate(R, st, S, ctx)
                cb, ib = c.min(1)
                best = torch.where((cb < best_c)[:, None, None], self.pick(S, ib), best)
                best_c = torch.minimum(best_c, cb)
            self.nom = best
            return best, {"iters": I, "rollouts": I * n, "cost": best_c}
        if m in ("mppi", "anneal"):
            kk = torch.arange(KNOTS, device="cuda", dtype=torch.float32)[:, None]
            for i in range(I):
                scale = sig if m == "mppi" else sig * torch.exp(-i / (PLAN["beta1"] * I) - (KNOTS - 1 - kk)
                                                                / (PLAN["beta2"] * KNOTS))
                S = self.constrain(self.nom[:, None] + self.noise(n, scale), u)
                S[:, 0] = self.constrain(self.nom[:, None], u)[:, 0]
                c = self.evaluate(R, st, S, ctx)
                lam = PLAN["lam_rel"] * (c.std(1, keepdim=True) + 1e-9)
                w = torch.softmax(-(c - c.min(1, keepdim=True).values) / lam, 1)
                self.nom = self.constrain(torch.einsum("pn,pnkj->pkj", w, S)[:, None], u)[:, 0]
            return self.nom, {"iters": I, "rollouts": I * n, "cost": c.min(1).values}
        if m in ("cem", "icem"):
            mean = nom0[:, 0].clone()
            std = torch.full_like(mean, sig)
            ne = max(2, int(round(PLAN["elite_frac"] * n)))
            best_c = torch.full((self.P,), float("inf"), device="cuda")
            best = mean.clone()
            for i in range(I):
                S = self.constrain(mean[:, None] + self.noise(n, std[:, None]), u)
                S[:, 0] = self.constrain(mean[:, None], u)[:, 0]
                if m == "icem" and i == 0 and self.elites is not None:
                    k_ = min(self.elites.shape[1], n - 1)
                    S[:, n - k_:] = self.constrain(self.elites[:, :k_], u)
                c = self.evaluate(R, st, S, ctx)
                idx = torch.argsort(c, 1)
                cb = c.gather(1, idx[:, :1])[:, 0]
                best = torch.where((cb < best_c)[:, None, None], self.pick(S, idx[:, 0]), best)
                best_c = torch.minimum(best_c, cb)
                el = S[self.ar[:, None], idx[:, :ne]]
                mom = PLAN["momentum"]
                mean = (1 - mom) * el.mean(1) + mom * mean
                std = ((1 - mom) * el.std(1) + mom * std).clamp(PLAN["sigma_min"], sig)
                if m == "icem":
                    self.elites = S[self.ar[:, None], idx[:, :PLAN["keep"]]].clone()
            self.nom = best if m == "icem" else self.constrain(mean[:, None], u)[:, 0]
            return self.nom, {"iters": I, "rollouts": I * n, "cost": best_c}
        if m in ("spider", "spider_wc"):
            prof = torch.logspace(math.log10(PLAN["sp_first"]), math.log10(PLAN["sp_last"]), KNOTS,
                                  device="cuda")[:, None]
            ns = torch.ones(n, 1, 1, device="cuda")
            ns[0] = 0.0
            n_ex = max(1, int(n * PLAN["sp_exploit"]))
            ns[n - n_ex:] = PLAN["sp_exploit_scale"]
            beta = PLAN["sp_final"] ** (1.0 / I)
            top = max(1, int(PLAN["sp_top"] * n))
            active = torch.ones(self.P, dtype=torch.bool, device="cuda")
            it = 0
            for i in range(I):
                S = self.constrain(self.nom[:, None] + self.noise(n, sig * prof) * ns * beta ** i, u)
                c = self.evaluate(R, st, S, ctx)
                r = -c / H
                tv, ti = r.topk(top, 1)
                z = (tv - tv.mean(1, keepdim=True)) / (tv.std(1, keepdim=True) + 1e-2)
                w = torch.softmax(z / PLAN["sp_temp"], 1)
                new = torch.einsum("pt,ptkj->pkj", w, S[self.ar[:, None], ti])
                self.nom = torch.where(active[:, None, None], self.constrain(new[:, None], u)[:, 0], self.nom)
                it += 1
                active &= (r.max(1).values - r[:, 0]) >= PLAN["sp_improve"]
                if not bool(active.any()):
                    break
            return self.nom, {"iters": it, "rollouts": it * n * self.G, "cost": c.min(1).values}
        raise ValueError(f"unknown method {m}")


# ------------------------------------------------------------------------------------------------ episodes
def ref_table(task):
    """The commanded rotation from the task's start after each period (n + 1, 4): uhas_sr2.Tracker.advance's
    integration of the commanded angular velocity."""
    q = torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda")
    out = [q]
    for i in range(task.n):
        w = task.w[i]
        ang = w.norm()
        dq = quat_from_axis_angle(w / ang.clamp_min(1e-12), ang * T.DT_C)
        q = quat_mul(dq, q)
        q = q / q.norm()
        out.append(q)
    return torch.stack(out)


def episode(B, chunk, method, R=None, policy=None, F=None, seed=0):
    """One batch of plant worlds through the episode: HOM to the task, then HOM, the policy or the planner. Returns
    the rows of hom_gpu.Batch.score, the usability metrics (inner_loop_usability.metrics) and the planner's record."""
    B.begin(chunk, record=True, hom=True)
    if float(B.cal_q.abs().max()) > 0:
        raise RuntimeError("calibration offsets in the plant: the planner's and the policy's targets bypass them")
    P, vw, k0 = B.W, B.vw, B.k
    task = UH.Task(B)
    plan_rec = {"walls": [], "iters": [], "rollouts": []}
    if method in METHODS:
        R.begin(chunk)
        r = R.G * R.n
        palm = vw["mocap_pos"][:, B.mo_palm].float().clone()
        p0 = vw["qpos"][:, B.qa:B.qa + 3].float() - palm
        q0 = vw["qpos"][:, B.qa + 3:B.qa + 7].float().clone()
        Qref = ref_table(task)
        ctx0 = {"palm": palm.repeat_interleave(r, 0), "p0": p0.repeat_interleave(r, 0)}
        mpc = MPC(method, P, R.lim.lo, R.lim.hi, R.n, R.G, seed)
        u = vw["ctrl"][:, B.act_t].float().clone()
    elif method == "uhas":
        from tensordict import TensorDict
        mounts = torch.tensor(np.array([[wd_["body_mounts"][f] for f in HG.FINGERS] for wd_ in chunk]),
                              dtype=torch.float32, device="cuda")
        Cc = U.SphereController(F, mounts)
        tr = UH.Tracker(P, "cuda")
        tr.reset(torch.arange(P, device="cuda"), B, Cc, F, mounts, B.n_out[:, 0].float())
    Q, Cm, A = [], [], []
    tau = 0
    with torch.inference_mode():
        while not B.done():
            if method == "hom":
                B.tick(None)
            elif method == "uhas":
                pol, cri = tr.observe(task, B, Cc, F, mounts)
                a = policy(TensorDict({"policy": pol, "critic": cri}, batch_size=[P], device="cuda")).clamp(-1, 1)
                B.tick(Cc.solve(a))
                tr.last_a = a
                tr.tau += 1
                tr.advance(task, B.scene.r)
            else:
                if tau % REPLAN == 0:
                    t0 = time.perf_counter()
                    i = torch.arange(tau + 1, tau + H + 1, device="cuda").clamp(max=task.n)
                    goal = quat_mul(Qref[i][None].expand(P, H, 4), q0[:, None].expand(P, H, 4))
                    knots, inf = mpc.plan(R, B.state(), u, dict(ctx0, goal=goal.repeat_interleave(r, 0)))
                    seg = torch.einsum("hk,pkj->phj", W_EXEC, knots)
                    torch.cuda.synchronize()
                    plan_rec["walls"].append(time.perf_counter() - t0)
                    plan_rec["iters"].append(inf["iters"])
                    plan_rec["rollouts"].append(inf["rollouts"])
                u = seg[:, tau % REPLAN]
                B.tick(u.reshape(P, 3, 3))
            Q.append(vw["qpos"][:, B.qadr_t].reshape(P, 9).float().clone())
            Cm.append(IU.targets(B).reshape(P, 9).clone())
            A.append(wp.to_torch(B.wd.actuator_force)[:, B.act_t].float().clone())
            tau += 1
    res = B.finish()
    M = IU.metrics(B, k0, torch.stack(Q), torch.stack(Cm), torch.stack(A), IU.Limits(B), B.prm.dt)
    return res, M, plan_rec


def row_key(r):
    return (r["controller"], r["x_sep"], r["y_sep"], r["grasp"], r["seed"], r["plant_mu"])


def cmd_run(a):
    import hom_landscape as L
    configure(a.replan, a.horizon, a.knots, a.iters)
    if (a.replan, a.horizon, a.knots, a.iters) != (10, 30, 3, 4):
        a.tag = f"_r{a.replan}h{a.horizon}k{a.knots}i{a.iters}" + a.tag
    prs = json.loads((ROWS / "mpc_pairs_twist.json").read_text())["pairs"]
    if a.limit_pairs:
        prs = prs[:a.limit_pairs]
    scene, worlds = worlds_of(prs, a.seeds)
    tag = "" if a.plant_mu == 1.0 else f"_mu{a.plant_mu:g}"
    out = ROWS / f"mpc_{a.method}_twist{tag}{a.tag}.jsonl"
    done = set()
    if out.exists():
        done = {row_key(json.loads(x)) for x in out.read_text().splitlines() if x.strip()}
    todo = [w for w in worlds if (a.method, *UH.xy_of(w["key"]), int(w["key"].get("grasp", 0)), w["seed"],
                                  a.plant_mu) not in done]
    print(f"{a.method}: {len(todo)} of {len(worlds)} worlds to run ({len(prs)} pairs, seeds {a.seeds}), plant friction "
          f"{a.plant_mu:g}", flush=True)
    if not todo:
        return 0
    Wc = min(a.chunk, len(todo))
    policy, ckpt, F, R = None, None, None, None
    B = None
    for c0 in range(0, len(todo), Wc):
        chunk = todo[c0:c0 + Wc]
        pad = chunk + [chunk[-1]] * (Wc - len(chunk))
        if B is None:
            B = UH.UBatch(scene, TASK, pad, T.DT_C, gen_seed=0)
            if a.plant_mu != 1.0:
                set_friction(B, np.full(Wc, a.plant_mu, np.float32))
            if a.method in METHODS:
                fr = PLAN["frictions"] if a.method == "spider_wc" else None
                n = a.n // len(fr) if fr else a.n
                R = Rollouts(scene, pad, n, fr)
            elif a.method == "uhas":
                F = U.Fingers(B.m, "cuda")
                policy, ckpt = UH.load_policy(UH.RUNS / a.run)
        t0 = time.time()
        res, M, pr = episode(B, pad, a.method, R, policy, F, seed=int(chunk[0]["seed"]) + c0)
        wall = time.time() - t0
        rows = []
        for i, (wd_, r) in enumerate(zip(chunk, res)):
            x, y = UH.xy_of(wd_["key"])
            row = {"controller": a.method, "x_sep": x, "y_sep": y, "grasp": int(wd_["key"].get("grasp", 0)),
                   "seed": wd_["seed"], "by": wd_["by"], "plant_mu": a.plant_mu, "J": L.track_score(TASK, r)}
            row.update({k: r.get(k) for k in IU.KEEP if k != "J"})
            row.update({k: round(float(M[k][i]), 4) for k in IU.METRICS})
            row.update({"chunk_wall_s": round(wall, 2), "chunk_worlds": Wc, "run": a.run if a.method == "uhas" else None,
                        "checkpoint": Path(ckpt).name if ckpt else None})
            if pr["walls"]:
                row.update({"replan_s_mean": round(float(np.mean(pr["walls"])), 4), "replans": len(pr["walls"]),
                            "rollouts_per_replan": round(float(np.mean(pr["rollouts"])), 1),
                            "iters_mean": round(float(np.mean(pr["iters"])), 2), "n": R.n, "groups": R.G,
                            "plan": {k: PLAN[k] for k in ("iters", "sigma")}, "replan": REPLAN, "H": H,
                            "knots": KNOTS})
            rows.append(row)
        with open(out, "a") as fh:
            for r_ in rows:
                fh.write(json.dumps(r_, default=float) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        js = [r_["J"] for r_ in rows]
        print(f"{a.method}: {c0 + len(chunk)} of {len(todo)} worlds; chunk {wall:.0f} s"
              + (f", replanning {np.mean(pr['walls']):.2f} s each ({len(pr['walls'])})" if pr["walls"] else "")
              + f"; J mean {np.mean(js):.3f}, held {np.mean([bool(r_['held']) for r_ in rows]):.2f}", flush=True)
    print(f"-> {out.relative_to(ROOT)}", flush=True)
    return 0


def cmd_pairs(a):
    prs = pairs(a.run, a.top)
    path = ROWS / "mpc_pairs_twist.json"
    path.write_text(json.dumps({"run": a.run, "top": a.top, "pairs": prs,
                                "rule": "the ten test layouts with HOM's best J (best grasp by the grid's mean over "
                                        "seeds 0-2) and the ten with UHAS's best J (best grasp by UHAS's J at seed 0)"},
                               indent=1))
    for p in prs:
        print(p)
    print(f"{len(prs)} pairs -> {path.relative_to(ROOT)}")
    return 0


def cmd_timing(a):
    """Wall time of a rollout period and of one replanning for a chunk of plant worlds."""
    configure(a.replan, a.horizon, a.knots, a.iters)
    prs = json.loads((ROWS / "mpc_pairs_twist.json").read_text())["pairs"]
    scene, worlds = worlds_of(prs, [0])
    chunk = (worlds * 4)[:a.chunk]
    B = UH.UBatch(scene, TASK, chunk, T.DT_C, gen_seed=0)
    out = []
    for n in a.n:
        R = Rollouts(scene, chunk, n)
        B.begin(chunk, record=False, hom=True)
        R.begin(chunk)
        vw = B.vw
        palm = vw["mocap_pos"][:, B.mo_palm].float()
        p0 = vw["qpos"][:, B.qa:B.qa + 3].float() - palm
        q0 = vw["qpos"][:, B.qa + 3:B.qa + 7].float()
        ctx = {"palm": palm.repeat_interleave(n, 0), "p0": p0.repeat_interleave(n, 0),
               "goal": q0[:, None].expand(len(chunk), H, 4).repeat_interleave(n, 0)}
        u = vw["ctrl"][:, B.act_t].float()
        mpc = MPC("mppi", len(chunk), R.lim.lo, R.lim.hi, n, 1, 0)
        with torch.inference_mode():
            mpc.plan(R, B.state(), u, ctx)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                mpc.plan(R, B.state(), u, ctx)
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 3
        per = dt / (PLAN["iters"] * H)
        row = {"plant_worlds": len(chunk), "n": n, "worlds": R.B.W, "iters": PLAN["iters"], "H": H,
               "replan": REPLAN, "replan_s": round(dt, 3),
               "ms_per_period": round(1e3 * per, 2), "us_per_world_period": round(1e6 * per / R.B.W, 2),
               "gpu": torch.cuda.get_device_name(0), "when": time.strftime("%Y-%m-%d %H:%M")}
        print(json.dumps(row), flush=True)
        out.append(row)
        del R
        torch.cuda.empty_cache()
    with open(ROWS / "mpc_timing_twist.jsonl", "a") as fh:
        for r_ in out:
            fh.write(json.dumps(r_) + "\n")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs")
    p.add_argument("--run", default=RUN)
    p.add_argument("--top", type=int, default=10)
    p = sub.add_parser("timing")
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--n", type=int, nargs="+", default=[128])
    p = sub.add_parser("run")
    p.add_argument("--method", required=True, choices=("hom", "uhas") + METHODS)
    p.add_argument("--run", default=RUN, help="the UHAS run (method uhas)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--plant-mu", type=float, default=1.0, help="the plant's sliding friction (the planners' model "
                   "keeps 1.0; spider_wc's spans 0.4-1.0)")
    p.add_argument("--chunk", type=int, default=16, help="plant worlds per batch")
    p.add_argument("--n", type=int, default=64, help="samples per plant world and iteration")
    p.add_argument("--limit-pairs", type=int, default=0, help="only the first pairs (smoke tests)")
    p.add_argument("--tag", default="", help="appended to the rows' file name")
    for q in (sub.choices["timing"], sub.choices["run"]):
        q.add_argument("--replan", type=int, default=10, help="periods between replannings")
        q.add_argument("--horizon", type=int, default=30, help="periods")
        q.add_argument("--knots", type=int, default=3)
        q.add_argument("--iters", type=int, default=4)
    a = ap.parse_args()
    return {"pairs": cmd_pairs, "timing": cmd_timing, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
