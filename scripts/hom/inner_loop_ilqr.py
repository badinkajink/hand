#!/usr/bin/env python3
"""iLQR on the SR2 hand in MuJoCo-Warp, method ilqr of inner_loop_mpc.py (owner, 2026-11-15: "did we also try
ilqr/ddp/feasibleddp on top of the gradient search?").

The plan is the nine servo targets, each held for STEP control periods (zero-order hold), so a horizon of H periods has
N = H / STEP steps. MuJoCo-Warp 3.6 has no derivatives. The dynamics are therefore linearized by forward finite
differences, as MJPC's iLQG does on the CPU with mjd_transitionFD. From the nominal state at the start of each step, one
step is run with the nominal targets, one with each tangent direction of the state perturbed and one with each target
perturbed. The tangent directions are the joint angles, the object's position, its orientation in its body frame and
the velocities (2 nv). All N (1 + 2 nv + 9) one-step rollouts of a plant world run in one batch.

Per replanning the shifted plan is rolled out from the plant's state, then ITERS iterations run:
  - the linearization A_t = dx_{t+1}/dx_t, B_t = dx_{t+1}/du_t in the tangent space;
  - the backward pass of iLQR (Li and Todorov 2004) with the regularization mu on Q_uu. The cost's rotation and
    position terms enter as residuals (Gauss-Newton), the servo-demand and joint-limit terms through their gradients, and
    the rate cost through a state augmented by the last target. V_zz is updated in Joseph form, the stage Hessian and
    V'_zz seen through the closed loop: a sum of positive semi-definite terms, so Q_uu >= 2 w I stays positive definite
    (the Schur-complement form lost definiteness on the hand, where A reaches about 60 and l_xx 1e6);
  - the forward pass: closed-loop rollouts u_t = clip(ubar_t + alpha k_t + K_t dz_t) at NA step sizes alpha (1 to
    2^-(NA-1)) in parallel. Each is scored with the planner's full cost (inner_loop_mpc.Rollouts.period_cost, with the
    pad and drop terms, plus the rate cost), and the cheapest becomes the nominal if it costs less than the nominal;
    else the nominal is kept. mu falls fourfold when the full step wins and rises tenfold when no step does.
The plant applies the nominal targets of the first REPLAN periods open loop, as the sampling planners do, or with the
feedback K_t dz_t at each step's start from the plant's state (IL fb).

The rate cost is matched to the knot planners' (inner_loop_mpc.MPC.rate_cost with three knots over 0.3 s): a constant
target rate costs the same under both.

Memory (11-15, RTX 4070 Ti SUPER, 16 GB): 6 hands per batch take 2.1 GB on the spheres and about 10 GB on the cylinders,
whose convex narrowphase sizes its buffers by the linearization batch (6 x 240 worlds). 16 hands fit on the spheres; on
the cylinders the first batch asks for 8.5 GB more and fails, and at 6 hands each further batch of one process adds
3.2 GB and fails. Run the cylinder tasks one batch per process: a loop of runs at --chunk 6, each resuming from the rows
already written (logs/20261115-ilqr/lane.txt, lines r20c_*).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sr2hand.rl.math import quat_conj, quat_from_axis_angle, quat_mul, quat_rotate, quat_to_rotvec, rotate_axis  # noqa: E402

# step: periods per step; na: line-search step sizes; mu0, mu_min, mu_max: the regularization; eps_*: finite-difference
# steps (joint and object angles rad, object position m, joint and angular velocities rad/s, object velocity m/s, targets
# rad); fb: the gains also in the plant; knot_gap_s: the knot planners' gap that the rate cost is matched to
IL = dict(step=5, na=8, mu0=1.0, mu_min=1e-4, mu_max=1e6, eps_q=1e-3, eps_p=2e-4, eps_w=1e-2, eps_v=2e-3, eps_u=1e-3,
          fb=False, knot_gap_s=0.15)


class TangentMap:
    """Differences and increments of MuJoCo states in the tangent space: hinge and slide coordinates as they are, the
    free joints' positions in the world frame and orientations in the body frame (MuJoCo's convention for qvel), then
    the velocities. A state is (qpos (..., nq), qvel (..., nv))."""

    def __init__(self, m):
        self.nq, self.nv = m.nq, m.nv
        lin_q, lin_d, self.quats = [], [], []
        for j in range(m.njnt):
            qa, da = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                lin_q += [qa, qa + 1, qa + 2]
                lin_d += [da, da + 1, da + 2]
                self.quats.append((qa + 3, da + 3))
            elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
                self.quats.append((qa, da))
            else:
                lin_q.append(qa)
                lin_d.append(da)
        self.lin_q, self.lin_d = lin_q, lin_d

    def diff(self, q, v, q0, v0):
        """(q, v) minus (q0, v0) in the tangent space, (..., 2 nv)."""
        d = q.new_zeros(q.shape[:-1] + (2 * self.nv,))
        d[..., self.lin_d] = q[..., self.lin_q] - q0[..., self.lin_q]
        for qa, da in self.quats:
            d[..., da:da + 3] = quat_to_rotvec(quat_mul(quat_conj(q0[..., qa:qa + 4]), q[..., qa:qa + 4]))
        d[..., self.nv:] = v - v0
        return d

    def plus_q(self, q0, d):
        """The positions q0 moved by the tangent increments d (..., at least nv entries; the first nv used)."""
        q = q0.clone(memory_format=torch.contiguous_format)
        q[..., self.lin_q] = q0[..., self.lin_q] + d[..., self.lin_d]
        for qa, da in self.quats:
            w = d[..., da:da + 3]
            ang = w.norm(dim=-1)
            qq = quat_mul(q0[..., qa:qa + 4], quat_from_axis_angle(w / ang.clamp_min(1e-12)[..., None], ang))
            q[..., qa:qa + 4] = qq / qq.norm(dim=-1, keepdim=True)
        return q

    def plus(self, q0, v0, d):
        return self.plus_q(q0, d), v0 + d[..., self.nv:]


def riccati(A, B, lx, lxx, du, gu, w, mu):
    """The backward pass of Gauss-Newton iLQR on the state z = (x, u_prev), for P problems of N steps:
    x_{t+1} = A_t x_t + B_t u_t in the tangent space, cost sum_{t>=1} l(x_t) + sum_t w |u_t - u_{t-1}|^2 + g(u_t).
    A (P, N, nx, nx), B (P, N, nx, nu); lx (P, N, nx) and lxx (P, N, nx, nx): l's gradient and Gauss-Newton Hessian at
    x_1..x_N; du (P, N, nu): the nominal's u_t - u_{t-1}; gu (P, N, nu): g's gradient; w: the rate weight; mu (P,): the
    regularization on Q_uu. The value function is updated in Joseph form. Returns k (P, N, nu), K (P, N, nu, nz) and a
    (P,) mask of the problems whose gains were not finite (set to zero)."""
    P, N, nx, nu = B.shape
    nz = nx + nu
    dt, dev = A.dtype, A.device
    Iu = torch.eye(nu, dtype=dt, device=dev)
    Vz = torch.zeros((P, nz), dtype=dt, device=dev)
    Vzz = torch.zeros((P, nz, nz), dtype=dt, device=dev)
    Vz[:, :nx], Vzz[:, :nx, :nx] = lx[:, N - 1], lxx[:, N - 1]
    k = torch.zeros((P, N, nu), dtype=dt, device=dev)
    K = torch.zeros((P, N, nu, nz), dtype=dt, device=dev)
    bad = torch.zeros(P, dtype=torch.bool, device=dev)
    Az = torch.zeros((P, nz, nz), dtype=dt, device=dev)
    Bz = torch.zeros((P, nz, nu), dtype=dt, device=dev)
    Bz[:, nx:] = Iu
    luz = torch.zeros((P, nu, nz), dtype=dt, device=dev)
    luz[:, :, nx:] = -2.0 * w * Iu
    for t in reversed(range(N)):
        Az[:, :nx, :nx] = A[:, t]
        Bz[:, :nx] = B[:, t]
        lz = torch.zeros((P, nz), dtype=dt, device=dev)
        lzz = torch.zeros((P, nz, nz), dtype=dt, device=dev)
        if t > 0:
            lz[:, :nx], lzz[:, :nx, :nx] = lx[:, t - 1], lxx[:, t - 1]
        lz[:, nx:] = -2.0 * w * du[:, t]
        lzz[:, nx:, nx:] = 2.0 * w * Iu
        lu = 2.0 * w * du[:, t] + gu[:, t]
        BzT = Bz.transpose(1, 2)
        luu = 2.0 * w * Iu
        Qu = lu + (BzT @ Vz[..., None])[..., 0]
        Quu = luu + BzT @ Vzz @ Bz
        Quz = luz + BzT @ Vzz @ Az
        Quu = 0.5 * (Quu + Quu.transpose(1, 2))
        L, info = torch.linalg.cholesky_ex(Quu + mu[:, None, None] * Iu)
        kt = -torch.cholesky_solve(Qu[..., None], L)[..., 0]
        Kt = -torch.cholesky_solve(Quz, L)
        ok = (info == 0) & torch.isfinite(kt).all(-1) & torch.isfinite(Kt).all((-1, -2))
        bad |= ~ok
        kt = torch.where(ok[:, None], kt, torch.zeros_like(kt))
        Kt = torch.where(ok[:, None, None], Kt, torch.zeros_like(Kt))
        # Joseph form: V_zz = [I; K]^T H [I; K] + (A + B K)^T V'_zz (A + B K), H the stage Hessian in (z, u)
        KtT, luzT = Kt.transpose(1, 2), luz.transpose(1, 2)
        Acl = Az + Bz @ Kt
        AclT = Acl.transpose(1, 2)
        Vz_n = (lz + (KtT @ lu[..., None])[..., 0] + (luzT @ kt[..., None])[..., 0] + (KtT @ (luu @ kt[..., None]))[..., 0]
                + (AclT @ (Vz + (Vzz @ (Bz @ kt[..., None]))[..., 0])[..., None])[..., 0])
        Vzz = lzz + luzT @ Kt + KtT @ luz + KtT @ luu @ Kt + AclT @ Vzz @ Acl
        Vzz = 0.5 * (Vzz + Vzz.transpose(1, 2))
        Vz = Vz_n
        k[:, t], K[:, t] = kt, Kt
    return k, K, bad


class Planner:
    """iLQR for the P plant worlds of a batch: a line-search batch of P x NA worlds and a linearization batch of
    P x N x (1 + 2 nv + 9) worlds, both inner_loop_mpc.Rollouts in the planner's scene and physics. M is the running
    inner_loop_mpc module, read for H, REPLAN, COST, T and Rollouts as configured."""

    def __init__(self, M, scene, chunk, iters):
        self.M, self.S, self.I = M, IL["step"], iters
        if M.H % self.S or M.REPLAN % self.S:
            raise SystemExit(f"ilqr: the step ({self.S} periods) must divide the horizon ({M.H}) and the replanning "
                             f"interval ({M.REPLAN})")
        self.P, self.N, self.NA = len(chunk), M.H // self.S, IL["na"]
        self.ls = M.Rollouts(scene, chunk, self.NA)
        B = self.ls.B
        m = B.m
        self.tm = TangentMap(m)
        nv = m.nv
        self.nx = 2 * nv
        self.mfd = 1 + self.nx + 9
        self.fd = M.Rollouts(scene, chunk, self.N * self.mfd)
        self.lim, self.q9 = self.ls.lim, self.ls.q9
        self.n, self.G = 1, 1                                    # the episode's context is per plant world
        qadr = np.asarray(B.qadr).reshape(-1)
        jid = [int(np.flatnonzero(m.jnt_qposadr == a)[0]) for a in qadr]
        self.d9 = torch.tensor(m.jnt_dofadr[jid], dtype=torch.long, device="cuda")
        self.qa, self.va = B.qa, B.va
        eps = np.full(self.nx, IL["eps_q"])
        eps[nv:] = IL["eps_w"]
        eps[self.va:self.va + 3] = IL["eps_p"]
        eps[nv + self.va:nv + self.va + 3] = IL["eps_v"]
        self.eps = torch.tensor(eps, dtype=torch.float64, device="cuda")
        self.E = torch.zeros((self.mfd, self.nx), device="cuda")
        self.E[1:1 + self.nx] = torch.diag(self.eps.float())
        self.Eu = torch.zeros((self.mfd, 9), device="cuda")
        self.Eu[1 + self.nx:] = IL["eps_u"] * torch.eye(9, device="cuda")
        self.alpha = torch.tensor([2.0 ** -i for i in range(self.NA)], dtype=torch.float64, device="cuda")
        s, dtc = math.radians(M.COST["rate_dps"]), M.T.DT_C
        self.w = M.COST["rate"] * (self.S * dtc / IL["knot_gap_s"]) / (self.S * dtc * s) ** 2
        self.hs = torch.arange(1, self.N + 1, device="cuda") * self.S - 1      # the period that ends each step
        self.ubar = self.K = None

    def begin(self, chunk):
        self.ls.begin(chunk)
        self.fd.begin(chunk)
        self.ubar = self.K = None
        self.mu = torch.full((self.P,), IL["mu0"], dtype=torch.float64, device="cuda")
        # per plant world: iterations, accepted steps, full steps (alpha 1), replannings, the summed relative decrease
        # of the plan's cost over a replanning, and Q_uu's that needed the fallback
        self.cnt = torch.zeros((self.P, 6), dtype=torch.float64, device="cuda")

    def stats(self, i):
        """Plant world i's record of the episode: the share of iterations whose line search lowered the cost, the share
        that took the full step, the mean relative decrease of the cost per replanning, and the replannings."""
        n, a, f, r, g, b = self.cnt[i].tolist()
        return {"accept": round(a / max(n, 1), 4), "full_step": round(f / max(n, 1), 4),
                "cost_drop": round(g / max(r, 1), 4), "replans": int(r), "fallback": int(b)}

    # ------------------------------------------------------------------------------------------- rollouts
    def rollout(self, st, cx, k=None, K=None):
        """Closed-loop rollouts of the line-search batch from the plant state st, the NA step sizes per plant world
        (each the nominal when k is None); cx the cost's context per world of the batch. Returns the costs (P, NA), the
        states at the steps' starts and the horizon's end (qpos (P, NA, N + 1, nq), qvel and warm start (P, NA, N + 1,
        nv)) and the applied targets (P, NA, N, 9)."""
        P, N, S, NA = self.P, self.N, self.S, self.NA
        R, B = self.ls, self.ls.B
        vw = B.vw
        R.load(st)
        warm = B.state().get("warm")
        c = torch.zeros(P * NA, device="cuda")
        drop = torch.zeros(P * NA, dtype=torch.bool, device="cuda")

        def rep(x):
            return x.repeat_interleave(NA, 0)
        ub, u_prev = rep(self.ubar), rep(self.u0)
        if k is not None:
            al = self.alpha.repeat(P)[:, None]
            kk, KK, xq, xv = rep(k), rep(K), rep(self.xq), rep(self.xv)
            upb = rep(torch.cat([self.u0[:, None], self.ubar[:, :-1]], 1))
        Xq, Xv, Xw, U = [], [], [], []
        for t in range(N + 1):
            Xq.append(vw["qpos"].clone())
            Xv.append(vw["qvel"].clone())
            Xw.append(warm.clone() if warm is not None else torch.zeros_like(Xv[-1]))
            if t == N:
                break
            u = ub[:, t]
            if k is not None:
                dz = torch.cat([self.tm.diff(Xq[-1], Xv[-1], xq[:, t], xv[:, t]), u_prev - upb[:, t]], -1).double()
                u = (u.double() + al * kk[:, t] + (KK[:, t] @ dz[..., None])[..., 0]).float()
            u = torch.maximum(torch.minimum(u, self.lim.hi), self.lim.lo)
            for s in range(S):
                vw["ctrl"][:, B.act_t] = u
                self.M.wp.capture_launch(B.graph)
                c += R.period_cost(u, cx, t * S + s, drop)
            c += self.w * ((u - u_prev) ** 2).sum(-1)
            u_prev = u
            U.append(u)
        c = torch.nan_to_num(c, nan=1e6, posinf=1e6)

        def stack(L):
            return torch.stack(L, 1).reshape(P, NA, len(L), -1)
        return c.reshape(P, NA), stack(Xq), stack(Xv), stack(Xw), stack(U)

    def linearize(self, st):
        """A (P, N, nx, nx) and B (P, N, nx, 9) by forward differences from the nominal states at the steps' starts
        (self.xq, xv, xw) with the nominal targets (self.ubar), float64."""
        P, N, S, Mf, nx = self.P, self.N, self.S, self.mfd, self.nx
        B = self.fd.B
        vw, St = B.vw, B.state()
        W = P * N * Mf
        q0 = self.xq[:, :N, None].expand(P, N, Mf, -1).contiguous()
        v0 = self.xv[:, :N, None].expand(P, N, Mf, -1).contiguous()
        q, v = self.tm.plus(q0, v0, self.E.expand(P, N, Mf, nx))
        St["qpos"].copy_(q.reshape(W, -1))
        St["qvel"].copy_(v.reshape(W, -1))
        if "warm" in St:
            St["warm"].copy_(self.xw[:, :N, None].expand(P, N, Mf, -1).reshape(W, -1))
        St["ctrl"].copy_(st["ctrl"].repeat_interleave(N * Mf, 0))
        St["mocap_pos"].copy_(st["mocap_pos"].repeat_interleave(N * Mf, 0))
        u = (self.ubar[:, :, None] + self.Eu).reshape(W, 9)
        for s in range(S):
            vw["ctrl"][:, B.act_t] = u
            self.M.wp.capture_launch(B.graph)
        q1, v1 = vw["qpos"].reshape(P, N, Mf, -1), vw["qvel"].reshape(P, N, Mf, -1)
        d = self.tm.diff(q1, v1, q1[:, :, :1].expand_as(q1), v1[:, :, :1].expand_as(v1)).double()
        A = (d[:, :, 1:1 + nx] / self.eps[None, None, :, None]).transpose(2, 3)
        Bm = (d[:, :, 1 + nx:] / IL["eps_u"]).transpose(2, 3)
        return A, Bm

    # ------------------------------------------------------------------------------------------- cost model
    def residuals(self, q, ctx):
        """The cost's rotation and position terms as residuals (P, N, 6) at the steps' ends, from the positions q
        (P, N, nq), float64: their squares are (rotation error / rot_deg)^2 and (distance / pos_mm)^2 of period_cost."""
        C, qa, hs = self.M.COST, self.qa, self.hs
        p = q[..., qa:qa + 3] - ctx["palm"][:, None].double()
        qo = q[..., qa + 3:qa + 7]
        if "aref" in ctx:
            rr = rotate_axis(qo, (0.0, 0.0, 1.0)) - ctx["aref"][:, hs].double()
        else:
            rr = quat_to_rotvec(quat_mul(qo, quat_conj(ctx["goal"][:, hs].double())))
        pt = p + quat_rotate(qo, ctx["pb"][:, None].double().expand_as(p)) if "pb" in ctx else p
        pref = ctx["pref"][:, hs].double() if "pref" in ctx else ctx["p0"][:, None].double()
        return torch.cat([rr / math.radians(C["rot_deg"]), (pt - pref) / (1e-3 * C["pos_mm"])], -1)

    def residual_jac(self, q, ctx, eps=1e-5):
        """The residuals and their Jacobian (P, N, 6, nx) in the tangent space (the object's six coordinates)."""
        q = q.double()
        r0 = self.residuals(q, ctx)
        J = q.new_zeros(r0.shape + (self.nx,))
        for d in range(self.va, self.va + 6):
            e = q.new_zeros(self.nx)
            e[d] = eps
            J[..., d] = (self.residuals(self.tm.plus_q(q, e), ctx) - r0) / eps
        return r0, J

    def demand_grad(self, u, q):
        """The servo-demand term's gradient over a step (P, N, 9) for the targets u (P, N, 9) and the joint angles at the
        steps' ends (positions q (P, N, nq))."""
        C, lim, kp = self.M.COST, self.lim, self.ls.kp.double()
        e = u.double() - q[..., self.q9].double()
        x = (kp * e.abs() / lim.tau.double() - C["tau_on"]) / (1.0 - C["tau_on"])
        on = ((x > 0) & (x < 3)).double()
        return C["tau"] * self.S * on * kp / lim.tau.double() / (1.0 - C["tau_on"]) * torch.sign(e)

    def band_grad(self, q):
        """The joint-limit term's gradient over a step in the tangent space (P, N, nx) at the positions q (P, N, nq)."""
        C, lim = self.M.COST, self.lim
        band = math.radians(C["q_band_deg"])
        qj = q[..., self.q9].double()
        a, b = qj - lim.lo.double(), lim.hi.double() - qj
        x = (band - torch.minimum(a, b)) / band
        on = ((x > 0) & (x < 1) & lim.limited).double()
        out = q.new_zeros(q.shape[:-1] + (self.nx,), dtype=torch.float64)
        out[..., self.d9] = C["q"] * self.S * on * (-torch.where(a < b, 1.0, -1.0) / band)
        return out

    # ------------------------------------------------------------------------------------------- planning
    def plan(self, R, st, u, ctx):
        """The targets of the next REPLAN periods (P, REPLAN, 9) from the plant state st; u (P, 9) the targets applied
        now; ctx the cost's context per plant world. R is unused (the planner holds its batches)."""
        P, N, S, NA = self.P, self.N, self.S, self.NA
        sh = self.M.REPLAN // S
        if self.ubar is None:
            self.ubar = u[:, None].expand(P, N, 9).clone()
        else:
            self.ubar = torch.cat([self.ubar[:, sh:], self.ubar[:, -1:].expand(P, sh, 9)], 1)
        self.u0 = u.clone()
        cx = {k_: v.repeat_interleave(NA, 0) for k_, v in ctx.items()}
        c, Xq, Xv, Xw, _U = self.rollout(st, cx)
        self.cost = c[:, -1]
        c_start = self.cost.double().clone()
        self.xq, self.xv, self.xw = Xq[:, -1], Xv[:, -1], Xw[:, -1]
        ar = torch.arange(P, device="cuda")
        n_ls, n_fd = 1, 0
        for _ in range(self.I):
            A, Bm = self.linearize(st)
            n_fd += 1
            r, Jr = self.residual_jac(self.xq[:, 1:], ctx)
            JrT = Jr.transpose(-1, -2)
            lx = 2.0 * S * (JrT @ r[..., None])[..., 0] + self.band_grad(self.xq[:, 1:])
            lxx = 2.0 * S * JrT @ Jr
            du = (self.ubar - torch.cat([self.u0[:, None], self.ubar[:, :-1]], 1)).double()
            gu = self.demand_grad(self.ubar, self.xq[:, 1:])
            k, K, bad = riccati(A, Bm, lx, lxx, du, gu, self.w, self.mu)
            self.cnt[:, 5] += bad.double()
            c, Xq, Xv, Xw, U = self.rollout(st, cx, k, K)
            n_ls += 1
            j = c.argmin(1)
            better = c[ar, j] < self.cost
            full = better & (j == 0)

            def keep(new, old):
                return torch.where(better.view((-1,) + (1,) * (new.dim() - 1)), new, old)
            self.cost = keep(c[ar, j], self.cost)
            self.xq, self.xv, self.xw = keep(Xq[ar, j], self.xq), keep(Xv[ar, j], self.xv), keep(Xw[ar, j], self.xw)
            self.ubar = keep(U[ar, j], self.ubar)
            self.K = keep(K, self.K) if self.K is not None else torch.where(better.view(-1, 1, 1, 1), K,
                                                                            torch.zeros_like(K))
            self.mu = torch.where(full, self.mu / 4.0, torch.where(better, self.mu, 10.0 * self.mu)).clamp(
                IL["mu_min"], IL["mu_max"])
            self.cnt[:, 0] += 1.0
            self.cnt[:, 1] += better.double()
            self.cnt[:, 2] += full.double()
        self.cnt[:, 3] += 1.0
        self.cnt[:, 4] += ((c_start - self.cost.double()) / c_start.clamp_min(1e-9)).clamp(min=0.0)
        rollouts = (n_ls * NA * self.M.H + n_fd * N * self.mfd * S) / self.M.H      # in rollouts of the horizon
        return self.ubar[:, :sh].repeat_interleave(S, 1), {"iters": self.I, "rollouts": rollouts, "cost": self.cost}

    def control(self, B, j):
        """The targets of period j of the replanning interval in the plant: the nominal, or with IL fb the nominal plus
        the gains on the plant's state at each step's start."""
        t = j // self.S
        if not IL["fb"] or self.K is None:
            return self.ubar[:, t]
        if j % self.S == 0:
            st = B.state()
            up_bar = self.u0 if t == 0 else self.ubar[:, t - 1]
            u_prev = self.u0 if t == 0 else self.u_fb
            dz = torch.cat([self.tm.diff(st["qpos"], st["qvel"], self.xq[:, t], self.xv[:, t]), u_prev - up_bar], -1)
            u = self.ubar[:, t].double() + (self.K[:, t] @ dz.double()[..., None])[..., 0]
            self.u_fb = torch.maximum(torch.minimum(u.float(), self.lim.hi), self.lim.lo)
        return self.u_fb
