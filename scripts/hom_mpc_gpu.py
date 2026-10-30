#!/usr/bin/env python3
"""Sampling MPC over the HOM controller's commands on MuJoCo-Warp, 2026-10-30.

The decision variables are HOM commands: per knot of KNOT periods, the object's reference angular velocity about the
vertical (deg/s) and a scale of the commanded pad forces. Each candidate plan runs in its own MuJoCo-Warp world under
the batched HOM controller (hom_gpu.Batch.control), which turns the commands into servo targets every 10 ms, so the
sampler searches a 2-D space per knot instead of the nine servo targets per step of the joint-space planner
(hom_tasks_mpc.py, pinned iCEM). The plant is a one-world batch of the same scene; at each replanning its full state
(MuJoCo-Warp data, contact forces, controller state) is copied into every planner world.

Task (twist, 40 mm sphere): from the HOM schedule's state after the lift (t = 2.0 s), turn the sphere about the vertical
as far as possible in TASK_S seconds while all three pads keep touching it. Cost of a plan over its horizon:
    - turned angle (deg) + W_LOST per period with a pad under 0.05 N + (drift / 3 mm)^2 per period
    + W_DROP if the centre falls 5 mm + W_F (scale - 1)^2 per knot.
The planner is a cross-entropy method over the knots (ITERS iterations of K samples, the ELITE best refit with
smoothing, the previous plan shifted by one knot kept as a sample). The baseline is the HOM schedule's 30 deg/s ramp
from the same state.

    .venv/bin/python scripts/hom_mpc_gpu.py run --call 280 --k 128      # MPC and baseline on grid layout 280
    .venv/bin/python scripts/hom_mpc_gpu.py timing --k 64 256 1024      # wall time per control period
Rows: docs/experiments/20261029-foundational_pose_review/rows/mpc_v4b/<name>.json
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
import warp as wp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_gpu as HG  # noqa: E402
import hom_landscape as L  # noqa: E402

T = HG.T
OUT = HG.ROWS / "mpc_v4b"
KNOT = 10                  # periods per knot (0.1 s)
H_KNOTS = 3                # horizon, knots (0.3 s)
ITERS, ELITE = 2, 12
TASK_S = 3.0               # s of task after the lift
W_LOST, W_DROP, W_F = 5.0, 200.0, 2.0
OMEGA = (0.0, 90.0)        # deg/s, the bounds of the commanded rate
FSCALE = (0.6, 1.6)        # bounds of the force scale
DATA = ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "time", "qacc_warmstart")
CTRL = ("theta_ref", "c1", "ap_x", "R_bm", "p_bm")


def world_of(task, d_mm, call, grasp, seed=0):
    """The batch world of one grid layout and grasp (hom_gpu's world dict) and its scene."""
    import fp_sr2_toy as FP
    shape = T.SHAPE[task]
    FP.set_sizes(**({"sph_d_mm": d_mm} if shape == "sphere" else {"cyl_d_mm": d_mm}))
    xs, ys, rec = L.grid_poses(task, d_mm)[call]
    g = (rec.get("grasps") or [rec])[grasp]
    six = FP.layout2(xs, ys)
    mxy = HG.mount_xy(six)
    if shape == "sphere":
        obj, _tg, n, _pc, _ax = FP.geometry("tripod", np.asarray(g["v"], float), FP.mounts6(six))
        setup = HG.tripod_setup(task, mxy, (np.asarray(obj["centre"], float),
                                            {f: np.asarray(n[f], float) for f in HG.FINGERS}), d_mm)
    else:
        setup = HG.pinch_setup(task, g["v"], six)
    scene = HG.Scene(shape, d_mm)
    bm, q_open = scene.prepare(mxy, setup)
    world = {"key": {"call": call, "grasp": grasp, "x_sep_mm": xs, "y_sep_mm": ys}, "seed": seed, "setup": setup,
             "body_mounts": bm, "q_open": q_open, "var": HG.variant_world("base", seed)}
    return scene, world


# The batched controller made free of host synchronisations so that one control period can be captured as a CUDA graph.
# It lives here until the GPU queues of 2026-10-30 finish; then Batch.control in hom_gpu.py takes these changes.

def solve3(Mm, rhs):
    """Batched 3x3 solves by the adjugate (no host sync, capturable)."""
    a, b, c = Mm[..., 0, 0], Mm[..., 0, 1], Mm[..., 0, 2]
    d, e, f = Mm[..., 1, 0], Mm[..., 1, 1], Mm[..., 1, 2]
    g, h, i = Mm[..., 2, 0], Mm[..., 2, 1], Mm[..., 2, 2]
    A_ = e * i - f * h
    B_ = -(d * i - f * g)
    C_ = d * h - e * g
    det = a * A_ + b * B_ + c * C_
    import torch as th
    inv = th.stack([th.stack([A_, -(b * i - c * h), b * f - c * e], -1),
                    th.stack([B_, a * i - c * g, -(a * f - c * d)], -1),
                    th.stack([C_, -(a * h - b * g), a * e - b * d], -1)], -2) / det[..., None, None]
    return (inv @ rhs[..., None])[..., 0]


class FastBatch(HG.Batch):
    """hom_gpu.Batch with control_task: Batch.control in the task phase (approach off, forces on, no pusher) with float
    masks for the fingers' roles, the tip indices on the GPU, closed-form 3 x 3 solves, Newton's method on the pad's
    s^(-1/2) - 1 with NEWTON_IT steps (surfaces.ellipsoid_closest) and the state updated in place."""
    NEWTON_IT = 6

    def reset_controller(self):
        super().reset_controller()
        th = self.torch
        ax, ay, az, azp = self.a_t.unbind(-1)
        self.a_halves = (th.stack([ax, ay, az]), th.stack([ax, ay, azp]))
        self.k_small = (int(th.argmin(self.a_halves[0])), int(th.argmin(self.a_halves[1])))
        self.follow_f = self.follow.to(self.f64)
        self.tip_t = th.tensor(self.tip, device=self.dev)
        self.graph_c = None

    def _ell_newton_phi(self, p, a, k):
        th = self.torch
        a2 = a * a
        pp = th.where(p.abs() < 1e-12, th.full_like(p, 1e-12), p)
        c = (a * pp) ** 2
        lo = -a2[k] + a[k] * pp[..., k].abs()
        hi = -a2[k] + (a * pp).norm(dim=-1)
        t = lo.clone()
        for _ in range(self.NEWTON_IT):
            d = a2 + t[..., None]
            s = (c / d ** 2).sum(-1)
            phi = s.rsqrt() - 1.0
            lo = th.where(phi < 0, t, lo)
            hi = th.where(phi < 0, hi, t)
            tn = t - phi / (s ** -1.5 * (c / d ** 3).sum(-1)).clamp_min(1e-300)
            ok = (tn > lo) & (tn < hi)
            t = th.where(ok, tn, 0.5 * (lo + hi))
        x = a2 * pp / (a2 + t[..., None])
        n = th.nn.functional.normalize(x / a2, dim=-1)
        dist = (pp - x).norm(dim=-1)
        return x, n, th.where(((pp / a) ** 2).sum(-1) < 1.0, -dist, dist)

    def _ell_closest(self, x):
        th = self.torch
        p = x - self.c_t
        qd, nd, sd_d = self._ell_newton_phi(p, self.a_halves[0], self.k_small[0])
        qp, np_, sd_p = self._ell_newton_phi(p, self.a_halves[1], self.k_small[1])
        ok_d, ok_p = qd[..., 2] <= 1e-12, qp[..., 2] >= -1e-12
        use_p = ok_p & (~ok_d | (sd_p.abs() < sd_d.abs()))
        q = th.where(use_p[..., None], qp, qd)
        n = th.where(use_p[..., None], np_, nd)
        return q + self.c_t, n, th.where(use_p, sd_p, sd_d)

    def control_task(self, meas, mv, V_obj):
        """Batch.control in the task phase (approach off, forces on, no pusher), with in-place state updates, float
        masks instead of boolean indexing and closed-form 3x3 solves: no host synchronisation."""
        th, f64, pr = self.torch, self.f64, self.prm
        dt = pr.dt
        W = self.W
        theta = meas["q"]
        xpos, xmat = mv["xpos"].to(f64), mv["xmat"].to(f64)
        pb, Rb = xpos[:, self.tip_t], xmat[:, self.tip_t]
        o, Ro = xpos[:, self.ob], xmat[:, self.ob]
        anc = mv["xanchor"].to(f64)[:, self.jid_t]
        axs = mv["xaxis"].to(f64)[:, self.jid_t]
        Fm, valid, cop = meas["F"], meas["valid"], meas["cop"]
        RbT = Rb.transpose(-1, -2)

        def to_tip(p):
            return th.einsum("wfij,wfj->wfi", RbT, p - pb)

        def to_world(p):
            return pb + th.einsum("wfij,wfj->wfi", Rb, p)

        oo = o[:, None, :].expand(-1, 3, -1)
        s1c, n1c, _ = self._tip_closest(to_tip(cop))
        p1c = to_world(s1c)
        n1w = th.einsum("wfij,wfj->wfi", Rb, n1c)
        p0c, n0c = self._obj_closest(cop, o, Ro)
        x1c = th.nn.functional.normalize(n1w - n0c, dim=-1)
        distc = ((p0c - p1c) * x1c).sum(-1)
        p1w, p0w, x1w, distw = self._witness(pb, Rb, o, Ro, to_tip, to_world)
        use_cop = (valid & (Fm >= self.cfmin[:, None]))[..., None]
        p1 = th.where(use_cop, p1c, p1w)
        x1 = th.where(use_cop, x1c, x1w)
        dist = th.where(use_cop[..., 0], distc, distw)
        ay = axs[:, :, 2]
        y = ay - (ay * x1).sum(-1, keepdim=True) * x1
        y = th.nn.functional.normalize(y, dim=-1)
        z = th.cross(x1, y, dim=-1)
        R1T = th.stack([x1, y, z], -1).transpose(-1, -2)
        V = V_obj[:, None, :] * self.follow_f[None, :, None]
        w_o = th.einsum("wij,wfj->wfi", Ro, V[..., :3])
        v_p1 = th.einsum("wij,wfj->wfi", Ro, V[..., 3:]) + th.cross(w_o, p1 - oo, dim=-1)
        Jr = axs.transpose(-1, -2)
        Jp = th.cross(axs, p1[:, :, None, :] - anc, dim=-1).transpose(-1, -2)
        A = th.cat([R1T @ Jr, R1T @ Jp], -2)
        b = -th.cat([th.einsum("wfij,wfj->wfi", R1T, w_o), th.einsum("wfij,wfj->wfi", R1T, v_p1)], -1)
        v_ref = th.zeros_like(b)
        force = self.force_on
        touching = (Fm >= self.cfmin[:, None]) | (dist <= 0.0)
        floop = (self.flg > 0)[:, None] & touching
        v_ref[..., 3] = v_ref[..., 3] + th.where(floop & (force > 0), self.flg[:, None] * (force - Fm), th.zeros_like(Fm))
        aw = (dist < pr.aw_dist) & (v_ref[..., 3] > 0) & ~floop
        v_ref[..., 3] = th.where(aw, th.zeros_like(dist), v_ref[..., 3])
        sw = th.sqrt(self.weights) * self.scale
        As = sw[None, :, :, None] * A
        rs = sw[None] * (v_ref - b)
        I3 = th.eye(3, dtype=f64, device=self.dev)
        Hm = As.transpose(-1, -2) @ As + (pr.w_damp + pr.w_healthy) * I3
        g = th.einsum("wfij,wfi->wfj", As, rs) + pr.w_healthy * pr.k_healthy * (self.healthy - theta)
        lb = th.maximum(th.full_like(theta, -pr.u_max), (self.lo_t - self.theta_ref) / dt)
        ub = th.minimum(th.full_like(theta, pr.u_max), (self.hi_t - self.theta_ref) / dt)
        lb, ub = th.minimum(lb, ub - 1e-12), th.maximum(ub, lb + 1e-12)
        P = self.pats
        fixed = (P > 0)[:, None, None, :]
        bound = th.where(P[:, None, None, :] == 1, lb[None], ub[None])
        Mm = th.where(fixed[..., None], I3.expand(27, 1, 1, 3, 3) * 1.0, Hm[None].expand(27, -1, -1, -1, -1))
        cc = th.where(fixed, bound, g[None].expand(27, -1, -1, -1))
        uu = solve3(Mm, cc)
        feas = ((uu >= lb[None] - 1e-9) & (uu <= ub[None] + 1e-9)).all(-1)
        obj = 0.5 * th.einsum("pwfi,wfij,pwfj->pwf", uu, Hm, uu) - (uu * g[None]).sum(-1)
        obj = th.where(feas, obj, th.full_like(obj, float("inf")))
        best = obj.argmin(0)
        u_sol = th.gather(uu, 0, best[None, ..., None].expand(1, -1, -1, 3))[0]
        u_sol = th.minimum(th.maximum(u_sol, lb), ub)
        tr = self.theta_ref + dt * u_sol
        tr = th.minimum(th.maximum(tr, theta - pr.lead), theta + pr.lead)
        self.theta_ref.copy_(th.minimum(th.maximum(tr, self.lo_t), self.hi_t))
        l1 = to_tip(p1)
        q1, n1, _ = self._tip_closest(self.c1)
        e = l1 - q1
        e_t = e - (e * n1).sum(-1, keepdim=True) * n1
        self.c1.copy_(self._tip_closest(q1 + pr.filter.k_min * dt * e_t)[0])
        s_act, n_act_l, _ = self._tip_closest(self.c1)
        p_act = to_world(s_act)
        n_act = th.einsum("wfij,wfj->wfi", Rb, n_act_l)
        Ja = th.cross(axs, p_act[:, :, None, :] - anc, dim=-1).transpose(-1, -2)
        gap = (dist - (n_act * th.einsum("wfij,wfj->wfi", Ja, self.theta_ref - theta)).sum(-1)).clamp(0.0, pr.gap_max)
        JJ = Ja @ Ja.transpose(-1, -2) + pr.dls ** 2 * I3
        step = solve3(JJ, gap[..., None] * n_act)
        imp = th.einsum("wfij,wfi->wfj", Ja, step) + th.einsum("wfij,wfi->wfj", Ja, force[..., None] * n_act) / self.kp_t
        offset = th.where((force > 0)[..., None], imp, th.zeros_like(imp))
        if pr.gravity_ff:
            offset = offset + self.gravity(mv["xipos"].to(f64), anc, axs) / self.kp_t
        return th.minimum(th.maximum(self.theta_ref + offset, self.lo_t), self.hi_t)

    def capture_control(self, meas, mv, V_obj):
        """Capture control_task as one CUDA graph on static inputs; returns (graph, static meas, static V, output)."""
        th = self.torch
        sm = {k: v.clone() for k, v in meas.items()}
        sV = V_obj.clone()
        state = (self.theta_ref.clone(), self.c1.clone())
        s = th.cuda.Stream()
        s.wait_stream(th.cuda.current_stream())
        with th.cuda.stream(s):
            for _ in range(3):
                self.control_task(sm, mv, sV)
        th.cuda.current_stream().wait_stream(s)
        g = th.cuda.CUDAGraph()
        with th.cuda.graph(g):
            out = self.control_task(sm, mv, sV)
        self.theta_ref.copy_(state[0])
        self.c1.copy_(state[1])
        return g, sm, sV, out


class Runner:
    """A batch stepped one control period at a time, with the object's reference twist and the pad forces given per
    world, or the HOM schedule's (schedule=True). With graph=True the task phase's control period runs as one captured
    CUDA graph of FastBatch.control_task; before it, and with graph=False, Batch.control runs eagerly."""

    def __init__(self, scene, task, world, n, graph=True):
        import torch
        self.th = torch
        self.use_graph, self.graph = graph, None
        self.B = B = FastBatch(scene, task, [world] * n, T.DT_C)
        B.reset_controller()
        B.sch = T.Schedule(task, world["setup"], T.PRM)
        self.vw = B.views(B.wd)
        B.mv = B.views(B.wdm)
        self.ft, self.fo, self.cp = (wp.to_torch(x) for x in (B.ftip, B.ftot, B.cops))
        self.f0 = B.force_on.clone()
        self.k, self.t, self.post_off = 0, 0.0, False

    def true_state(self):
        B, th = self.B, self.th
        qpos = self.vw["qpos"].to(B.f64)
        F = self.ft.to(B.f64)
        cop = self.cp.to(B.f64) / F.clamp_min(1e-12)[..., None]
        cop = th.where((F > 0)[..., None], cop, th.zeros_like(cop))
        return {"q": qpos[:, B.qadr_t], "pos": qpos[:, B.qa:B.qa + 3], "quat": qpos[:, B.qa + 3:B.qa + 7], "cop": cop,
                "F": F}

    def period(self, w_z=None, fscale=None, schedule=False):
        """One period at self.t. w_z: the commanded angular velocity about the vertical per world (rad/s, tensor W);
        fscale: the force scale per world (tensor W)."""
        B, th, t = self.B, self.th, self.t
        if t >= T.T_LIFT and not self.post_off:
            self.vw["mocap_pos"][:, B.mo_post, 2] -= 1.0
            self.post_off = True
        meas = B.measure(self.k, self.true_state())
        mv = B.mirror(meas)
        Rd = B._quat_mat(meas["quat_d"])
        W = B.W
        w = th.zeros((W, 3), dtype=B.f64, device=B.dev)
        if schedule:
            w_ref, _s, _th, _ph = B.schedule(t)
            w[:] = th.tensor(w_ref, dtype=B.f64, device=B.dev)
        elif w_z is not None:
            w[:, 2] = w_z
        V = th.cat([th.einsum("wji,wj->wi", Rd, w), th.zeros_like(w)], -1)
        B.force_on.copy_(self.f0 * (fscale[:, None] if fscale is not None else 1.0))
        if self.use_graph and not schedule and t >= T.T_LIFT + 0.4 - 1e-9:
            if self.graph is None:
                self.graph = B.capture_control(meas, mv, V)
            g, sm, sV, q_cmd = self.graph
            for k_, v in meas.items():
                sm[k_].copy_(v)
            sV.copy_(V)
            g.replay()
        else:
            q_cmd = B.control(t, meas, mv, V, 0.0)
        self.vw["ctrl"][:, B.act_t] = (q_cmd - B.cal_q).reshape(W, 9).float()
        wp.capture_launch(B.graph)
        self.k += 1
        self.t += B.prm.dt

    def snapshot(self):
        B = self.B
        snap = {f: wp.to_torch(getattr(B.wd, f)).clone() for f in DATA if hasattr(B.wd, f)}
        snap.update(ft=self.ft.clone(), fo=self.fo.clone(), cp=self.cp.clone(),
                    ctrl_state={c: (None if getattr(B, c) is None else getattr(B, c).clone()) for c in CTRL},
                    buf={k_: v.clone() for k_, v in B.buf.items()}, k=self.k, t=self.t, post_off=self.post_off)
        return snap

    def load(self, snap):
        """Every world of this runner takes world 0 of the snapshot."""
        B = self.B
        for f in DATA:
            if f in snap:
                dst = wp.to_torch(getattr(B.wd, f))
                dst.copy_(snap[f][:1].expand_as(dst))
        for name, src in (("ft", snap["ft"]), ("fo", snap["fo"]), ("cp", snap["cp"])):
            dst = getattr(self, name)
            dst.copy_(src[:1].expand_as(dst))
        for c, v in snap["ctrl_state"].items():                 # in place: a captured graph holds these tensors
            cur = getattr(B, c)
            if v is None:
                setattr(B, c, None)
            elif cur is not None and cur.shape[1:] == v.shape[1:]:
                cur.copy_(v[:1].expand_as(cur))
            else:
                setattr(B, c, v[:1].expand(B.W, *v.shape[1:]).clone())
        for k_, v in snap["buf"].items():
            B.buf[k_] = v[:, :1].expand(-1, B.W, *v.shape[2:]).clone()
        self.k, self.t, self.post_off = snap["k"], snap["t"], snap["post_off"]

    def yaw_rate(self):
        """The object's angular velocity about the world vertical (rad/s, per world)."""
        B = self.B
        R = self.vw["xmat"][:, B.ob].to(B.f64)
        wl = self.vw["qvel"][:, B.va + 3:B.va + 6].to(B.f64)
        return self.th.einsum("wij,wj->wi", R, wl)[:, 2]


def plan_cost(P, plans, snap):
    """Run every plan (K x H_KNOTS x 2: deg/s, force scale) from the snapshot in the planner runner; the cost of each."""
    th = P.th
    P.load(snap)
    B = P.B
    W = B.W
    pos0 = P.true_state()["pos"].clone()
    A = th.tensor(plans, dtype=B.f64, device=B.dev)
    turned = th.zeros(W, dtype=B.f64, device=B.dev)
    lost = th.zeros(W, dtype=B.f64, device=B.dev)
    drift = th.zeros(W, dtype=B.f64, device=B.dev)
    drop = th.zeros(W, dtype=th.bool, device=B.dev)
    for p in range(H_KNOTS * KNOT):
        a = A[:, p // KNOT]
        P.period(w_z=th.deg2rad(a[:, 0]), fscale=a[:, 1])
        turned += th.rad2deg(P.yaw_rate()) * B.prm.dt
        st = P.true_state()
        lost += (P.ft.to(B.f64) < 0.05).any(1).to(B.f64)
        dp = (st["pos"] - pos0).norm(dim=1)
        drift += (dp / 0.003) ** 2
        drop |= (pos0[:, 2] - st["pos"][:, 2]) > 0.005
    cost = -turned + W_LOST * lost + drift + W_DROP * drop.to(B.f64) + W_F * ((A[:, :, 1] - 1.0) ** 2).sum(1)
    return cost.cpu().numpy(), turned.cpu().numpy(), lost.cpu().numpy()


def cem(P, snap, mean, std, rng, k):
    """ITERS iterations of the cross-entropy method over the knots from the plant's snapshot; the best plan, its
    cost terms and the refit mean."""
    lo, hi = np.array([OMEGA[0], FSCALE[0]]), np.array([OMEGA[1], FSCALE[1]])
    best = None
    for _ in range(ITERS):
        plans = np.clip(mean[None] + std[None] * rng.normal(size=(k, H_KNOTS, 2)), lo, hi)
        plans[0] = mean
        if best is not None:
            plans[1] = best[0]
        cost, turned, lost = plan_cost(P, plans, snap)
        order = np.argsort(cost)
        if best is None or cost[order[0]] < best[1]:
            best = (plans[order[0]].copy(), float(cost[order[0]]), float(turned[order[0]]), float(lost[order[0]]))
        el = plans[order[:ELITE]]
        mean = 0.7 * el.mean(0) + 0.3 * mean
        std = np.maximum(0.7 * el.std(0) + 0.3 * std, np.array([2.0, 0.03]))
    return best, mean


def run_plant(scene, task, world, mpc_k=None, seed=0, graph=True):
    """The plant from the open pose: the HOM schedule up to the end of the lift, then for TASK_S seconds either the
    schedule's 30 deg/s ramp (mpc_k None) or the MPC with mpc_k samples. Returns the trace and the timing."""
    G = Runner(scene, task, world, 1, graph)
    t_task = T.T_LIFT + 0.4
    while G.t < t_task - 1e-9:
        G.period(schedule=True)
    P = Runner(scene, task, world, mpc_k, graph) if mpc_k else None
    rng = np.random.default_rng(seed)
    mean = np.tile([30.0, 1.0], (H_KNOTS, 1))
    std = np.tile([20.0, 0.2], (H_KNOTS, 1))
    th = G.th
    trace, plans, walls = [], [], []
    yaw = 0.0
    n_per = int(round(TASK_S / G.B.prm.dt))
    ramp_t0 = 1.3                      # the schedule's ramp starts 1.3 s into the task (after the two steps and rest)
    for p in range(n_per):
        if mpc_k and p % KNOT == 0:
            w0 = time.perf_counter()
            snap = G.snapshot()
            best, mean = cem(P, snap, mean, std, rng, mpc_k)
            th.cuda.synchronize()
            walls.append(time.perf_counter() - w0)
            plans.append({"t": round(G.t, 3), "knots": best[0].round(3).tolist(), "cost": round(best[1], 3),
                          "turned": round(best[2], 2), "lost": best[3]})
            act = best[0][0]
            mean = np.concatenate([mean[1:], mean[-1:]])
        if mpc_k:
            G.period(w_z=th.tensor([math.radians(act[0])], dtype=G.B.f64, device=G.B.dev),
                     fscale=th.tensor([act[1]], dtype=G.B.f64, device=G.B.dev))
        else:
            G.t_sched = G.t
            G.period(w_z=th.tensor([math.radians(30.0)], dtype=G.B.f64, device=G.B.dev))
        wz = float(G.yaw_rate()[0])
        yaw += math.degrees(wz) * G.B.prm.dt
        F = G.ft[0].cpu().numpy()
        trace.append({"t": round(G.t, 3), "yaw_deg": round(yaw, 3), "w_dps": round(math.degrees(wz), 2),
                      "F": F.round(3).tolist(), "F_link": (G.fo[0] - G.ft[0]).cpu().numpy().round(3).tolist(),
                      "z_mm": round(float(G.true_state()["pos"][0, 2]) * 1e3, 3)})
    three = [r["yaw_deg"] for r in trace if min(r["F"]) > 0.05]
    del ramp_t0
    return {"trace": trace, "plans": plans, "plan_wall_s": walls,
            "yaw_end_deg": trace[-1]["yaw_deg"], "yaw_three_pads_max_deg": max(three) if three else 0.0,
            "dropped": trace[-1]["z_mm"] < trace[0]["z_mm"] - 15.0}


def cmd_run(a):
    import torch
    scene, world = world_of(a.task, a.d, a.call, a.grasp)
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, k in (("hom_ramp", None), (f"mpc_k{a.k}", a.k)):
        t0 = time.time()
        r = run_plant(scene, a.task, world, k, graph=a.graph)
        r["wall_s"] = round(time.time() - t0, 2)
        res[name] = r
        pw = r["plan_wall_s"]
        print(f"{name}: yaw {r['yaw_end_deg']:.1f} deg, on three pads {r['yaw_three_pads_max_deg']:.1f} deg, dropped "
              f"{r['dropped']}, wall {r['wall_s']} s" + (f", replanning {np.median(pw):.2f} s for {KNOT * T.DT_C:.2f} s "
                                                        f"of control ({KNOT * T.DT_C / np.median(pw):.3f} x real time)"
                                                        if pw else ""), flush=True)
    out = {"task": a.task, "d_mm": a.d, "call": a.call, "grasp": a.grasp, "k": a.k, "graph": a.graph, "knot_s": KNOT * T.DT_C,
           "horizon_knots": H_KNOTS, "iters": ITERS, "elite": ELITE, "task_s": TASK_S,
           "gpu": torch.cuda.get_device_name(0), "when": time.strftime("%Y-%m-%d %H:%M"), "runs": res}
    path = OUT / f"twist_call{a.call}_g{a.grasp}_k{a.k}{'' if a.graph else '_eager'}.json"
    path.write_text(json.dumps(out))
    print("wrote", path.relative_to(ROOT), flush=True)
    return 0


def cmd_timing(a):
    """Wall time of one control period of the planner batch for each world count, from the state after the lift."""
    import torch
    scene, world = world_of(a.task, a.d, a.call, a.grasp)
    G = Runner(scene, a.task, world, 1)
    while G.t < T.T_LIFT + 0.4 - 1e-9:
        G.period(schedule=True)
    snap = G.snapshot()
    out = []
    for k in a.k:
        P = Runner(scene, a.task, world, k, graph=a.graph)
        P.load(snap)
        w = torch.full((k,), math.radians(30.0), dtype=P.B.f64, device=P.B.dev)
        for _ in range(5):
            P.period(w_z=w)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 50
        for _ in range(n):
            P.period(w_z=w)
        torch.cuda.synchronize()
        ms = 1e3 * (time.perf_counter() - t0) / n
        parts = {}
        B = P.B
        meas = B.measure(P.k, P.true_state())
        mv = B.mirror(meas)
        V = torch.zeros((k, 6), dtype=B.f64, device=B.dev)
        parts_fn = [("physics", lambda: wp.capture_launch(B.graph)), ("mirror", lambda: B.mirror(meas)),
                    ("controller_eager", lambda: B.control(P.t, meas, mv, V, 0.0)),
                    ("measure", lambda: B.measure(P.k, P.true_state()))]
        if P.graph is not None:
            parts_fn.append(("controller_graph", P.graph[0].replay))
        for name, fn in parts_fn:
            fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            parts[name] = round(1e3 * (time.perf_counter() - t1) / 20, 3)
        out.append({"k": k, "ms_per_period": round(ms, 3), "us_per_world_period": round(1e3 * ms / k, 2),
                    "ms_parts": parts})
        print(out[-1], flush=True)
        del P
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "timing.json").write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "rows": out,
                                                 "when": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    return 0


def cmd_figure(a):
    """The run's twist angle, the MPC's commands and the pad forces against the HOM ramp, for the HOM review page."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    d = json.loads((OUT / f"twist_call{a.call}_g{a.grasp}_k{a.k}.json").read_text())
    runs = d["runs"]
    fig, axs = plt.subplots(3, 1, figsize=(8.2, 8.4), sharex=True)
    for name, r, c in (("HOM ramp, 30 deg/s", runs["hom_ramp"], "#e6731a"), (f"MPC, {a.k} samples", runs[f"mpc_k{a.k}"], "k")):
        t = np.array([x["t"] for x in r["trace"]]) - (T.T_LIFT + 0.4)
        yaw = np.array([x["yaw_deg"] for x in r["trace"]])
        F = np.array([x["F"] for x in r["trace"]])
        axs[0].plot(t, yaw, color=c, lw=1.6, label=name)
        lost = np.flatnonzero(F.min(1) <= 0.05)
        if len(lost):
            axs[0].plot(t[lost[0]], yaw[lost[0]], "o", color=c, ms=8, mfc="none", mew=1.6)
        for k, f in enumerate(("thumb", "index", "middle")):
            axs[2].plot(t, F[:, k], color=("#1f77b4", "#d62728", "#2ca02c")[k], lw=1.2, ls="-" if c == "k" else "--",
                        label=f"{f}" if c == "k" else None)
    axs[0].set_ylabel("twist of the sphere (deg)")
    axs[0].legend(loc="lower right", frameon=False)
    axs[0].text(0.01, 0.92, "circle: first period with a pad under 0.05 N", transform=axs[0].transAxes, fontsize=9)
    P = runs[f"mpc_k{a.k}"]["plans"]
    tp = np.array([x["t"] for x in P]) - (T.T_LIFT + 0.4)
    kn = np.array([x["knots"][0] for x in P])
    axs[1].step(tp, kn[:, 0], where="post", color="k", lw=1.4, label="commanded rate (deg/s)")
    axs[1].axhline(30.0, color="#e6731a", lw=1.2, ls="--", label="the ramp's rate")
    ax2 = axs[1].twinx()
    ax2.step(tp, kn[:, 1], where="post", color="#9467bd", lw=1.2)
    ax2.set_ylabel("force scale", color="#9467bd")
    ax2.set_ylim(0.5, 1.7)
    axs[1].set_ylabel("rate (deg/s)")
    axs[1].legend(loc="upper right", frameon=False)
    axs[2].set_ylabel("pad force (N)")
    axs[2].set_xlabel("time from the task's start (s)")
    axs[2].legend(loc="upper left", frameon=False, ncol=3, title="solid: MPC; dashed: ramp", title_fontsize=9)
    fig.tight_layout()
    out = ROOT / "docs/experiments/20261029-hom_control_review/media" / f"20261030-mpc_twist_call{a.call}.png"
    fig.savefig(out, dpi=100)
    print("wrote", out.relative_to(ROOT), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "timing", "figure"):
        p = sub.add_parser(name)
        p.add_argument("--task", default="twist", choices=("twist",))
        p.add_argument("--d", type=float, default=40.0)
        p.add_argument("--call", type=int, default=280)
        p.add_argument("--grasp", type=int, default=0)
        p.add_argument("--k", type=int, nargs="+" if name == "timing" else None, default=[64, 256] if name == "timing"
                       else 64)
        p.add_argument("--eager", dest="graph", action="store_false", help="Batch.control without the CUDA graph")
    a = ap.parse_args()
    return {"run": cmd_run, "timing": cmd_timing, "figure": cmd_figure}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
