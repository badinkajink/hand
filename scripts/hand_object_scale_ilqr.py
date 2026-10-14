#!/usr/bin/env python3
"""Gradient-based MPC for the scale-study turn: iterative LQR (Gauss-Newton) over 20 ms control intervals, with the
dynamics and contact-sensor derivatives taken by finite differences of short MuJoCo rollouts (2026-10-13; owner: "if we
maintain the three contacts, maybe gradient based methods can work?").

Decision variables: the nine servo targets held constant over each of T = horizon / dt_plan intervals (25 over 0.5 s).
State: MuJoCo's tangent state (nv positions, nv velocities; 30). Cost: sum over intervals of |r_t|^2 with residuals
  tilt     sqrt(w_tilt / 2) (u - e_z)                            the marked axis toward vertical; |r|^2 = w_tilt (1 - u_z)
  pos      sqrt(w_pos) (p - p_ref) / 10 mm
  contact  sqrt(w_contact) softplus(k (F_min - F_f) / F_min) / k   per finger, F_f from the planner model's contact sensor
  rate     sqrt(w_rate) softplus(k (|omega| - omega_max) / omega_max) / k
  smooth   sqrt(w_du_step) (u_t - u_{t-1})                        u_{-1} is the command applied now
and a terminal tilt and position residual weighted w_term. Per replanning: one rollout of the current plan; for every
interval, 1 + 2 nv + nu rollouts of one interval (the plan and one-sided perturbations of the tangent state and the
targets), all in one mujoco.rollout call; a backward pass with Levenberg-Marquardt damping; a closed-loop forward pass
at step sizes 1, 0.5, 0.25 and 0.1, run in threads; the best that lowers the cost becomes the plan. The plan shifts one
interval per replanning.
"""
from __future__ import annotations

import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hand_object_scale_mpc as M  # noqa: E402
import hand_object_scale_sim as S  # noqa: E402

ILQR_DEFAULTS = dict(eps_q=1e-4, eps_v=1e-3, eps_u=1e-3, soft_k=5.0, w_du_step=2.0, mu0=1e-2, alphas=(1.0, 0.5, 0.25, 0.1))


def softplus(z):
    return np.where(z > 30, z, np.log1p(np.exp(np.minimum(z, 30))))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class ILQRPlanner:
    def __init__(self, plant, prm, nthread, rng=None):
        from mujoco import rollout
        mj = plant.mj
        self.mj, self.plant, self.prm = mj, plant, dict(ILQR_DEFAULTS, **prm)
        self.m = m = M.sensor_model(plant)
        post = m.geom("post").id
        m.geom_contype[post] = m.geom_conaffinity[post] = 0
        if prm.get("pad_filter", 0.0) > 0:
            pads = np.array([g for g in range(m.ngeom) if "_pad" in (m.geom(g).name or "")])
            tool = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == plant.tb])
            m.geom_contype[pads], m.geom_conaffinity[pads] = 2, 0
            m.geom_conaffinity[tool] = m.geom_conaffinity[tool] | 2
        self.pool = rollout.Rollout(nthread=nthread)
        self.datas = [mj.MjData(m) for _ in range(nthread)]
        self.fwd = [mj.MjData(m) for _ in range(len(self.prm["alphas"]))]
        self.spec = mj.mjtState.mjSTATE_FULLPHYSICS
        self.nstate = mj.mj_stateSize(m, self.spec)
        self.dt = float(m.opt.timestep)
        self.nsub = int(round(prm["dt_plan"] / self.dt))
        self.T = int(round(prm["horizon"] / prm["dt_plan"]))
        self.nv, self.nq = m.nv, m.nq
        self.act = np.asarray(plant.act)
        self.nu = len(self.act)
        self.lo, self.hi = m.actuator_ctrlrange[self.act, 0], m.actuator_ctrlrange[self.act, 1]
        self.qa, self.va = int(plant.qa), int(plant.va)
        self.U = None
        self.mu = self.prm["mu0"]
        self.x0 = np.empty(self.nstate)
        self.ctrl_tpl = plant.d.ctrl.copy()

    # ------------------------------------------------------------------ state helpers
    def q_of(self, x):
        return x[1:1 + self.nq]

    def v_of(self, x):
        return x[1 + self.nq:1 + self.nq + self.nv]

    def perturb(self, x, i, eps):
        """Full-physics state x moved by eps along tangent direction i (0..nv-1 positions, nv..2nv-1 velocities)."""
        y = x.copy()
        if i < self.nv:
            dq = np.zeros(self.nv)
            dq[i] = eps
            q = y[1:1 + self.nq].copy()
            self.mj.mj_integratePos(self.m, q, dq, 1.0)
            y[1:1 + self.nq] = q
        else:
            y[1 + self.nq + i - self.nv] += eps
        return y

    def tangent_diff(self, xa, xb):
        """Tangent difference xb - xa (2 nv)."""
        dq = np.zeros(self.nv)
        self.mj.mj_differentiatePos(self.m, dq, 1.0, self.q_of(xa), self.q_of(xb))
        return np.concatenate([dq, self.v_of(xb) - self.v_of(xa)])

    # ------------------------------------------------------------------ residuals
    def pose_res(self, q, p_ref, w_scale=1.0):
        prm = self.prm
        p = q[self.qa:self.qa + 3]
        w_, x_, y_, z_ = q[self.qa + 3:self.qa + 7] / np.linalg.norm(q[self.qa + 3:self.qa + 7])
        axis = np.array([2 * (x_ * z_ + w_ * y_), 2 * (y_ * z_ - w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)])
        r_t = math.sqrt(w_scale * prm["w_tilt"] / 2.0) * (axis - [0.0, 0.0, 1.0])
        r_p = math.sqrt(w_scale * prm["w_pos"]) * (p - p_ref) / M.SIGMA_POS
        return np.concatenate([r_t, r_p])

    def stage_res(self, x, F, u, u_prev, p_ref):
        """Residuals of one interval at its start state x, with finger forces F (after its first step) and target u."""
        prm, k = self.prm, self.prm["soft_k"]
        r = [self.pose_res(self.q_of(x), p_ref, 2.0)]
        if prm.get("w_contact", 0.0) > 0:
            r.append(math.sqrt(2.0 * prm["w_contact"]) * softplus(k * (prm["f_min"] - F) / prm["f_min"]) / k)
        if prm.get("w_rate", 0.0) > 0:
            om = np.linalg.norm(x[1 + self.nq + self.va + 3:1 + self.nq + self.va + 6])
            wm = math.radians(prm["rate_max_deg"])
            r.append(math.sqrt(2.0 * prm["w_rate"]) * softplus(np.array([k * (om - wm) / wm])) / k)
        r.append(math.sqrt(prm["w_du_step"]) * (u - u_prev))
        return np.concatenate(r)

    def forces(self, sens):
        """Per-finger contact force magnitude from the three contact sensors (9 numbers)."""
        return np.linalg.norm(sens.reshape(-1, len(S.FINGERS), 3), axis=-1)

    # ------------------------------------------------------------------ rollouts
    def nominal(self, U):
        """States at every interval start (T + 1) and finger forces after each interval's first step (T)."""
        ctrl = np.repeat(np.tile(self.ctrl_tpl, (self.T, 1)), self.nsub, axis=0)[None]
        ctrl[0, :, self.act] = np.repeat(U, self.nsub, axis=0)
        st, se = self.pool.rollout(self.m, self.datas, self.x0[None], ctrl, nstep=self.T * self.nsub)
        X = np.concatenate([self.x0[None], st[0, self.nsub - 1::self.nsub]])
        Fs = self.forces(se[0, ::self.nsub]) if self.m.nsensordata else np.zeros((self.T, 3))
        return X, Fs

    def derivatives(self, X, U):
        """A (T, 2nv, 2nv), B (T, 2nv, nu) of each interval map, and dF/dx (T, 3, 2nv), dF/du (T, 3, nu) of the
        finger forces after the interval's first step, by one-sided finite differences."""
        prm, nv, nu, T = self.prm, self.nv, self.nu, self.T
        npert = 1 + 2 * nv + nu
        init = np.empty((T * npert, self.nstate))
        ctrl = np.tile(self.ctrl_tpl, (T * npert, self.nsub, 1))
        for t in range(T):
            b = t * npert
            init[b] = X[t]
            for i in range(2 * nv):
                init[b + 1 + i] = self.perturb(X[t], i, prm["eps_q"] if i < nv else prm["eps_v"])
            init[b + 1 + 2 * nv:b + npert] = X[t]
            ctrl[b:b + npert, :, self.act] = U[t]
            for j in range(nu):
                ctrl[b + 1 + 2 * nv + j, :, self.act[j]] += prm["eps_u"]
        st, se = self.pool.rollout(self.m, self.datas, init, ctrl, nstep=self.nsub)
        A = np.empty((T, 2 * nv, 2 * nv))
        B = np.empty((T, 2 * nv, nu))
        FX = np.zeros((T, 3, 2 * nv))
        FU = np.zeros((T, 3, nu))
        for t in range(T):
            b = t * npert
            xe0 = st[b, -1]
            F0 = self.forces(se[b, 0])[0] if self.m.nsensordata else np.zeros(3)
            for i in range(2 * nv):
                e = prm["eps_q"] if i < nv else prm["eps_v"]
                A[t, :, i] = self.tangent_diff(xe0, st[b + 1 + i, -1]) / e
                if self.m.nsensordata:
                    FX[t, :, i] = (self.forces(se[b + 1 + i, 0])[0] - F0) / e
            for j in range(nu):
                B[t, :, j] = self.tangent_diff(xe0, st[b + 1 + 2 * nv + j, -1]) / prm["eps_u"]
                if self.m.nsensordata:
                    FU[t, :, j] = (self.forces(se[b + 1 + 2 * nv + j, 0])[0] - F0) / prm["eps_u"]
        return A, B, FX, FU

    def res_jac(self, x, F, FX, FU, u, u_prev, p_ref):
        """Residual r and its Jacobians wrt the tangent state (2nv) and the targets (nu) for one interval."""
        prm, nv, k = self.prm, self.nv, self.prm["soft_k"]
        r = self.stage_res(x, F, u, u_prev, p_ref)
        n_pose = 6
        J_x = np.zeros((len(r), 2 * nv))
        J_u = np.zeros((len(r), self.nu))
        eps = 1e-6
        for i in range(nv):      # pose residuals depend on positions only
            J_x[:n_pose, i] = (self.pose_res(self.q_of(self.perturb(x, i, eps)), p_ref, 2.0) - r[:n_pose]) / eps
        row = n_pose
        if prm.get("w_contact", 0.0) > 0:
            z = k * (prm["f_min"] - F) / prm["f_min"]
            g = math.sqrt(2.0 * prm["w_contact"]) * sigmoid(z) * (-1.0 / prm["f_min"])
            J_x[row:row + 3] = g[:, None] * FX
            J_u[row:row + 3] = g[:, None] * FU
            row += 3
        if prm.get("w_rate", 0.0) > 0:
            w = x[1 + self.nq + self.va + 3:1 + self.nq + self.va + 6]
            om = np.linalg.norm(w) + 1e-12
            wm = math.radians(prm["rate_max_deg"])
            g = math.sqrt(2.0 * prm["w_rate"]) * sigmoid(k * (om - wm) / wm) / wm
            J_x[row, nv + self.va + 3:nv + self.va + 6] = g * w / om
            row += 1
        J_u[row:row + self.nu] = math.sqrt(prm["w_du_step"]) * np.eye(self.nu)
        return r, J_x, J_u

    def total_cost(self, X, Fs, U, u_now, p_ref):
        c = 0.0
        for t in range(self.T):
            r = self.stage_res(X[t], Fs[t], U[t], U[t - 1] if t else u_now, p_ref)
            c += float(r @ r)
        rT = self.pose_res(self.q_of(X[-1]), p_ref, self.prm["w_term"])
        return c + float(rT @ rT)

    def forward(self, alpha, X, U, kff, K, u_now, p_ref, d):
        """Closed-loop forward pass at step size alpha on MjData d; its states, forces, targets and cost."""
        mj = self.mj
        mj.mj_setState(self.m, d, self.x0, self.spec)
        Xn = [self.x0.copy()]
        Fn, Un = [], []
        x = self.x0.copy()
        for t in range(self.T):
            dx = self.tangent_diff(X[t], x)
            u = np.clip(U[t] + alpha * kff[t] + K[t] @ dx, self.lo, self.hi)
            d.ctrl[:] = self.ctrl_tpl
            d.ctrl[self.act] = u
            for s in range(self.nsub):
                mj.mj_step(self.m, d)
                if s == 0:
                    Fn.append(self.forces(d.sensordata.copy())[0] if self.m.nsensordata else np.zeros(3))
            mj.mj_getState(self.m, d, x, self.spec)
            if not np.all(np.isfinite(x)):
                return None
            Xn.append(x.copy())
            Un.append(u)
        Xn, Fn, Un = np.array(Xn), np.array(Fn), np.array(Un)
        return Xn, Fn, Un, self.total_cost(Xn, Fn, Un, u_now, p_ref)

    # ------------------------------------------------------------------ one replanning
    def plan(self, t_now, p_ref, z0):
        mj, prm, nv, nu, T = self.mj, self.prm, self.nv, self.nu, self.T
        d = self.plant.d
        u_now = d.ctrl[self.act].copy()
        if self.U is None:
            self.U = np.tile(u_now, (T, 1))
        else:
            self.U = np.concatenate([self.U[1:], self.U[-1:]])
        mj.mj_getState(self.plant.m, d, self.x0, self.spec)
        X, Fs = self.nominal(self.U)
        c0 = self.total_cost(X, Fs, self.U, u_now, p_ref)
        A, B, FX, FU = self.derivatives(X, self.U)
        # terminal
        rT = self.pose_res(self.q_of(X[-1]), p_ref, prm["w_term"])
        JT = np.zeros((6, 2 * nv))
        for i in range(nv):
            JT[:, i] = (self.pose_res(self.q_of(self.perturb(X[-1], i, 1e-6)), p_ref, prm["w_term"]) - rT) / 1e-6
        Vx, Vxx = JT.T @ rT, JT.T @ JT
        kff = np.zeros((T, nu))
        K = np.zeros((T, nu, 2 * nv))
        for t in range(T - 1, -1, -1):
            r, Jx, Ju = self.res_jac(X[t], Fs[t], FX[t], FU[t], self.U[t], self.U[t - 1] if t else u_now, p_ref)
            Qx = Jx.T @ r + A[t].T @ Vx
            Qu = Ju.T @ r + B[t].T @ Vx
            Qxx = Jx.T @ Jx + A[t].T @ Vxx @ A[t]
            Quu = Ju.T @ Ju + B[t].T @ Vxx @ B[t] + self.mu * np.eye(nu)
            Qux = Ju.T @ Jx + B[t].T @ Vxx @ A[t]
            try:
                L = np.linalg.cholesky(Quu)
            except np.linalg.LinAlgError:
                self.mu = min(self.mu * 10, 1e6)
                return self._exec(), c0, c0
            kff[t] = -np.linalg.solve(Quu, Qu)
            K[t] = -np.linalg.solve(Quu, Qux)
            Vx = Qx + K[t].T @ Quu @ kff[t] + K[t].T @ Qu + Qux.T @ kff[t]
            Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]
            Vxx = 0.5 * (Vxx + Vxx.T)
            del L
        with ThreadPoolExecutor(len(self.prm["alphas"])) as ex:
            res = list(ex.map(lambda ad: self.forward(ad[0], X, self.U, kff, K, u_now, p_ref, ad[1]),
                              zip(self.prm["alphas"], self.fwd)))
        res = [r for r in res if r is not None]
        best = min(res, key=lambda r: r[3]) if res else None
        if best is not None and best[3] < c0:
            self.U = best[2]
            self.mu = max(self.mu / 2, 1e-6)
            c1 = best[3]
        else:
            self.mu = min(self.mu * 10, 1e6)
            c1 = c0
        return self._exec(), c1, c0

    def _exec(self):
        return np.tile(self.U[0], (self.nsub, 1))
