#!/usr/bin/env python3
"""MuJoCo-Warp rollouts for the scale-study MPC: the planner of hand_object_scale_mpc.py with its candidate plans
rolled out as parallel GPU worlds instead of CPU threads, so that a replanning can afford hundreds of samples.

GPUPlanner keeps every update rule of hand_object_scale_mpc.Planner (ps, mppi, cem, icem, anneal) and replaces
`evaluate`: the plant's state (float64 on the CPU) is broadcast to `nworld` worlds of the planner's model (contact
sensors, no post) in float32; one CUDA graph of `cost_every` steps sets each world's servo targets from a device array
before every step and records the cylinder's pose, angular velocity and the three contact sensors after the block; the
records come back to the host for the cost of hand_object_scale_mpc.Planner.cost. Contact buffers: 1,024 contacts and
2,048 constraint rows per world (1 mm pads write their broadphase candidates into the contact buffer; see
feedback_mjw_broadphase_and_contact_buffer).

  .venv/bin/python scripts/hand_object_scale_mpc_gpu.py check --cell cylinder:25:0.125 --nworld 64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp  # noqa: F401  (module level: the kernels' string annotations are evaluated against these globals)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hand_object_scale_mpc as M  # noqa: E402
import hand_object_scale_sim as S  # noqa: E402

NCONMAX, NJMAX = 1024, 2048


def _kernels():
    import warp as wp

    @wp.kernel
    def set_ctrl(U: wp.array3d(dtype=float), k: wp.array(dtype=int), act: wp.array(dtype=int),
                 ctrl: wp.array2d(dtype=float)):
        w, j = wp.tid()
        ctrl[w, act[j]] = U[k[0], w, j]

    @wp.kernel
    def bump(k: wp.array(dtype=int)):
        k[0] = k[0] + 1

    @wp.kernel
    def record(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float), sens: wp.array2d(dtype=float),
               r: wp.array(dtype=int), qa: int, wa: int, ns: int,
               rq: wp.array3d(dtype=float), rw: wp.array3d(dtype=float), rs: wp.array3d(dtype=float)):
        w = wp.tid()
        i = r[0]
        for c in range(7):
            rq[i, w, c] = qpos[w, qa + c]
        for c in range(3):
            rw[i, w, c] = qvel[w, wa + c]
        for c in range(ns):
            rs[i, w, c] = sens[w, c]

    return set_ctrl, bump, record


class GPUPlanner(M.Planner):
    """hand_object_scale_mpc.Planner whose rollouts run on MuJoCo-Warp; `nworld` worlds per evaluation."""

    def __init__(self, plant, prm, nthread, rng, nworld):
        import mujoco_warp as mjw
        import warp as wp
        super().__init__(plant, prm, 1, rng)
        self.mjw, self.wp = mjw, wp
        m = self.m
        # float32 never meets the CPU scene's 1e-10 tolerance, so the solver would run all 100 iterations every step
        m.opt.tolerance = float(prm.get("gpu_tol", 1e-6))
        m.opt.iterations = int(prm.get("gpu_iter", 20))
        m.opt.ls_iterations = int(prm.get("gpu_ls_iter", 10))
        self.nworld = nworld
        self.block = int(prm["cost_every"])
        if self.nh % self.block:
            raise ValueError("the horizon must be a whole number of cost samples")
        self.nrec = self.nh // self.block
        self.wm = mjw.put_model(m)
        # the CAD hand's link spheres overflowed the default 64 contact matches per sensor (2026-10-14: wrong per-finger
        # forces in the cost, then a segfault in an icem_gpu run)
        self.wm.opt.contact_sensor_maxmatch = max(int(self.wm.opt.contact_sensor_maxmatch), int(prm.get("maxmatch", 256)))
        self.wd = mjw.make_data(m, nworld=nworld, nconmax=NCONMAX, njmax=NJMAX)
        self.set_ctrl, self.bump, self.record = _kernels()
        self.U = wp.zeros((self.nh, nworld, len(plant.act)), dtype=float)
        self.k_dev = wp.zeros(1, dtype=int)
        self.r_dev = wp.zeros(1, dtype=int)
        self.act_dev = wp.array(np.asarray(plant.act, np.int32), dtype=int)
        self.ns = int(m.nsensordata)
        self.rq = wp.zeros((self.nrec, nworld, 7), dtype=float)
        self.rw = wp.zeros((self.nrec, nworld, 3), dtype=float)
        self.rs = wp.zeros((self.nrec, nworld, max(1, self.ns)), dtype=float)
        self.qa_m = int(plant.qa)
        self.wa_m = int(plant.va) + 3
        mjw.forward(self.wm, self.wd)
        # compile every kernel before the capture (a module cannot load inside a graph capture)
        wp.launch(self.set_ctrl, dim=(nworld, len(plant.act)), inputs=[self.U, self.k_dev, self.act_dev, self.wd.ctrl])
        wp.launch(self.bump, dim=1, inputs=[self.k_dev])
        wp.launch(self.record, dim=nworld, inputs=[self.wd.qpos, self.wd.qvel, self.wd.sensordata, self.r_dev,
                                                   self.qa_m, self.wa_m, self.ns, self.rq, self.rw, self.rs])
        mjw.step(self.wm, self.wd)
        self.k_dev.zero_()
        self.r_dev.zero_()
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            for _ in range(self.block):
                wp.launch(self.set_ctrl, dim=(nworld, len(plant.act)), inputs=[self.U, self.k_dev, self.act_dev,
                                                                              self.wd.ctrl])
                mjw.step(self.wm, self.wd)
                wp.launch(self.bump, dim=1, inputs=[self.k_dev])
            wp.launch(self.record, dim=nworld, inputs=[self.wd.qpos, self.wd.qvel, self.wd.sensordata, self.r_dev,
                                                       self.qa_m, self.wa_m, self.ns, self.rq, self.rw, self.rs])
            wp.launch(self.bump, dim=1, inputs=[self.r_dev])
        self.graph = cap.graph
        self.overflow = 0

    def _load_state(self):
        wp, d = self.wp, self.plant.d
        nw = self.nworld
        self.wd.qpos.assign(np.tile(d.qpos, (nw, 1)).astype(np.float32))
        self.wd.qvel.assign(np.tile(d.qvel, (nw, 1)).astype(np.float32))
        self.wd.ctrl.assign(np.tile(d.ctrl, (nw, 1)).astype(np.float32))
        self.wd.time.assign(np.full(nw, d.time, np.float32))
        if hasattr(self.wd, "qacc_warmstart"):
            self.wd.qacc_warmstart.assign(np.tile(d.qacc_warmstart, (nw, 1)).astype(np.float32))
        self.k_dev.zero_()
        self.r_dev.zero_()

    def evaluate(self, samples, p_ref, z0):
        wp = self.wp
        n = len(samples)
        if n > self.nworld:
            raise ValueError(f"{n} samples for {self.nworld} worlds")
        pad = np.concatenate([samples, np.repeat(samples[:1], self.nworld - n, 0)]) if n < self.nworld else samples
        ctrl = np.einsum("hk,nkj->nhj", self.W, pad)
        self.U.assign(np.ascontiguousarray(ctrl.transpose(1, 0, 2)).astype(np.float32))
        self._load_state()
        for _ in range(self.nrec):
            wp.capture_launch(self.graph)
        wp.synchronize()
        q = self.rq.numpy().transpose(1, 0, 2)[:n].astype(float)
        w = self.rw.numpy().transpose(1, 0, 2)[:n].astype(float)
        s = self.rs.numpy().transpose(1, 0, 2)[:n].astype(float) if self.ns else None
        c = self.cost_records(q, w, s, samples, p_ref, z0)
        return c, ctrl[:n]

    def cost_records(self, q, w, s, samples, p_ref, z0):
        """Planner.cost on the recorded samples (q: (n, nrec, 7) cylinder pose, w: angular velocity, s: sensors)."""
        import math
        prm = self.prm
        p, x, y = q[..., :3], q[..., 4], q[..., 5]
        nq = np.linalg.norm(q[..., 3:7], axis=-1)
        x, y = x / nq, y / nq
        uz = 1.0 - 2.0 * (x * x + y * y)
        run = prm["w_tilt"] * (1.0 - uz) + prm["w_pos"] * np.sum((p - p_ref) ** 2, -1) / M.SIGMA_POS ** 2 \
            + prm["w_drop"] * (p[..., 2] < z0 - 0.015)
        if prm.get("w_rate", 0.0) > 0:
            om = np.linalg.norm(w, axis=-1)
            wm = math.radians(prm["rate_max_deg"])
            run = run + prm["w_rate"] * np.maximum(0.0, om - wm) ** 2 / wm ** 2
        if prm.get("w_contact", 0.0) > 0 and s is not None:
            F = np.linalg.norm(s.reshape(s.shape[0], s.shape[1], len(S.FINGERS), 3), axis=-1)
            run = run + prm["w_contact"] * np.clip((prm["f_min"] - F) / prm["f_min"], 0.0, 1.0).sum(-1)
        run = np.where(np.isfinite(run), run, 1e6)
        c = run[:, :-1].sum(1) + prm["w_term"] * run[:, -1]
        return c + prm["w_du"] * np.sum(np.diff(samples, axis=1) ** 2, (1, 2))


def cmd_check(a) -> int:
    """GPU and CPU costs of the same candidate plans from one gripped state, and the GPU time per evaluation."""
    S.PLANT = a.plant
    shape, dd, t = a.cell.split(":")
    dd, t = float(dd), float(t)
    cell, _, _ = next(c for c in S.cell_list("diag") if c[1] == shape and c[2] == dd and c[0].get("t") == t)
    ci = M.hom_best(Path(a.hom_rows))[(cell["tag"], shape, dd)]["cand"]
    plant, meta, geo, sc, gg = M.setup(cell, shape, dd, ci)
    grip = np.array([meta["grip"][f][k] for f in S.FINGERS for k in range(3)])
    plant.reset(0)
    plant.set_targets(grip)
    plant.advance(0.8)
    prm = dict(M.DEFAULTS, **M.PLANNER_DEFAULTS, **M.CONTROLLED, nsample=a.nworld, horizon=0.5, pad_filter=float(a.pad_filter))
    for k in ("nsample", "knots", "cost_every", "iters", "keep", "log_every"):
        prm[k] = int(prm[k])
    rng = np.random.default_rng(0)
    cpu = M.Planner(plant, prm, 12, rng)
    gpu = GPUPlanner(plant, prm, 1, np.random.default_rng(0), a.nworld)
    s0 = plant.state()
    p_ref = s0["p"] + [0.0, 0.0, 0.005]
    z0 = float(s0["p"][2])
    cpu.nominal = np.tile(plant.d.ctrl[plant.act], (len(cpu.knot_t), 1))
    samples = cpu.constrain(cpu.nominal[None] + cpu.noise(a.nworld, prm["sigma"]), plant.d.ctrl[plant.act])
    plant.mj.mj_getState(plant.m, plant.d, cpu.x0, cpu.spec)
    t0 = time.perf_counter()
    c_cpu, _ = cpu.evaluate(samples, p_ref, z0)
    t_cpu = time.perf_counter() - t0
    gpu.evaluate(samples, p_ref, z0)                     # first call compiles
    t0 = time.perf_counter()
    c_gpu, _ = gpu.evaluate(samples, p_ref, z0)
    t_gpu = time.perf_counter() - t0
    r = float(np.corrcoef(c_cpu, c_gpu)[0, 1])
    rank = float(np.corrcoef(np.argsort(np.argsort(c_cpu)), np.argsort(np.argsort(c_gpu)))[0, 1])
    out = {"cell": a.cell, "nworld": a.nworld, "pad_filter": a.pad_filter, "cpu_s_12threads": round(t_cpu, 3), "gpu_s": round(t_gpu, 3),
           "cost_pearson": round(r, 4), "cost_spearman": round(rank, 4),
           "cost_cpu_median": round(float(np.median(c_cpu)), 3), "cost_gpu_median": round(float(np.median(c_gpu)), 3),
           "best_same": int(np.argmin(c_cpu)) == int(np.argmin(c_gpu))}
    print(json.dumps(out), flush=True)
    path = M.OUT / "planners" / "gpu_check.jsonl"
    with open(path, "a") as fh:
        fh.write(json.dumps(out) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--cell", default="cylinder:25:0.125")
    c.add_argument("--nworld", type=int, default=64)
    c.add_argument("--pad-filter", type=int, default=1)
    c.add_argument("--plant", default="kp0.72_kv0_fr0.18_fl0_dp0.0144")
    c.add_argument("--hom-rows", default=str(M.OUT / "hom_postoff_servo/turn.jsonl"))
    a = ap.parse_args()
    return {"check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
