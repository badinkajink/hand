#!/usr/bin/env python3
"""Sampling model-predictive control of the in-hand turn on the cells of the hand-object scale study, a second
controller beside the HOM turn of hand_object_scale_sim.py.

Scene, grasp candidate, plant, grip phase and held test are those of hand_object_scale_sim.turn_rollout with post_off:
the grip command for 0.8 s, the run ending at 6.3 s (HOM's longest run: squeeze, a 90 deg turn at 30 deg/s and the
2 s hold), held = the object less than 20 mm below its post-grip height with at least two fingers each pressing with at
least its weight. The post's collisions end at `t_post_off`, by default 0.8 s, when MPC takes over (HOM loses the post
at 1.3 s, when its turn begins; 2026-10-13: with the post kept to 1.3 s, MPC turned the cylinder 20-58 deg before
1.3 s, some runs with under 1 N of finger force, so the post could carry the early turn).

From 0.8 s the controller replans every `dt_plan` by predictive sampling (Howell et al. 2022, the planner of MuJoCo
MPC). The nominal plan is a piecewise-linear spline of the nine joint targets with `knots` knots over `horizon`;
`nsample` plans, the time-shifted nominal and noisy copies of it (Gaussian, `sigma` rad, clipped to the servo
ranges), are rolled out with mujoco.rollout on the plant's model without the post, and the cheapest becomes the
nominal. The cost
of a rollout, sampled every 10 ms:

    w_tilt (1 - u_z)                 u = the object's marked axis (body z); 0 when vertical, 1 when horizontal
  + w_pos |p - p_ref|^2 / (10 mm)^2  p_ref = the object centre after the grip, raised 5 mm (HOM's lift)
  + w_drop [p_z < z_0 - 15 mm]
  the last sample weighted w_term times, plus w_du times the squared knot-to-knot change of the joint targets.

  .venv/bin/python scripts/hand_object_scale_mpc.py run --only cylinder:16:0.375 --seeds 0 --out-dir /tmp/x
  .venv/bin/python scripts/hand_object_scale_mpc.py run --cells diag --shapes cylinder --seeds 0 1 2 --shard 0/2
Rows: docs/experiments/20261013-scale_turn_controllers/mpc/turn.jsonl, one fsynced line per rollout; resumable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import hand_object_scale_sim as S  # noqa: E402
import hom_turn3 as H  # noqa: E402

OUT = ROOT / "docs/experiments/20261013-scale_turn_controllers"
if S.HAND == "cad_v1":       # SCALE_HAND=cad_v1: every default output and HOM-row input of the CAD-hand rerun
    OUT = ROOT / "docs/experiments/20261014-scale_turn_cad_v1"
HOM_ROWS = OUT / "hom_postoff/turn.jsonl"
DEFAULTS = dict(nsample=32, horizon=0.3, knots=3, dt_plan=0.02, sigma=0.08, w_tilt=1.0, w_pos=1.0, w_drop=10.0,
                w_term=10.0, w_du=1.0, lift_mm=5.0, t_grip=0.8, t_squeeze=0.5, t_end=6.3, cost_every=10, t_post_off=0.8,
                w_contact=0.0, f_min=0.5, w_rate=0.0, rate_max_deg=60.0)
# The controlled variant (owner 2026-09-16: a reorientation must be controlled and end held on three pads; the first
# MPC grid swung the cylinder about a two-finger pinch at a median peak of 308 deg/s and ended 67 of 95 held runs on
# two fingers): --set w_contact=1 w_rate=1 adds, per 10 ms sample, w_contact * sum_f clip((f_min - F_f) / f_min, 0, 1)
# with F_f each finger's net contact force on the cylinder (a MuJoCo contact sensor per tip body in the planner's model)
# and w_rate * relu(|omega| - rate_max)^2 / rate_max^2 on the cylinder's angular speed.
CONTROLLED = dict(w_contact=1.0, w_rate=1.0)
SIGMA_POS = 0.010
# Planner variants (2026-10-13, owner: "can we see if we can improve this smoothness/jitteriness ... compare predictive
# sampling to CEM, iCEM, and ... annealed sampling"). Every variant spends `nsample` rollouts per replanning; a method
# with `iters` inner iterations draws nsample // iters per iteration.
#   ps      predictive sampling: the cheapest of the shifted nominal and its noisy copies (Howell et al. 2022)
#   mppi    one MPPI step: the exp(-(J - J_min) / lam) weighted average, lam = lam_rel * std(J) (0.03: at 0.1 and 0.5 the
#           average barely moved, 0.6-1.1 deg in 0.5 s on the 25 mm cell, against 11.8 deg at 0.03)
#   cem     cross-entropy: per iteration the elite_frac cheapest set the mean and the standard deviation (momentum
#           `momentum`, floor sigma_min); the mean is executed
#   icem    CEM with knot-correlated noise (AR(1), ar_alpha), the `keep` best elites of the last replanning shifted into
#           the first population, and the best sample executed (Pinneri et al. 2020; mjpc_icra2026 planners/icem)
#   anneal  DIAL-MPC annealed sampling as in mujoco_mpc (branch sampling) planners/annealed_sampling: iters stages,
#           knot noise sigma * exp(-i / (beta1 iters) - (K - 1 - h) / (beta2 K)), an MPPI update after each stage
# pin0=1 fixes the first knot to the command applied now, so a replanning cannot make the command jump.
PLANNER_DEFAULTS = dict(method="ps", iters=1, lam_rel=0.03, elite_frac=0.15, momentum=0.1, ar_alpha=0.7, keep=2,
                        beta1=0.5, beta2=0.5, sigma_min=0.005, pin0=0.0, log_every=10)


def hom_best(path: Path = HOM_ROWS) -> dict:
    """Per cell, the candidate grasp with the largest median held HOM turn (the landscape's rule) and its turns."""
    by = defaultdict(lambda: defaultdict(list))
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("status") == "ok" and r.get("model", "pads") == "pads":
            by[(r["tag"], r["shape"], r["d_mm"])][r["cand"]].append(r["held_turn_deg"])
    out = {}
    for k, cc in by.items():
        ci = max(sorted(cc), key=lambda c: float(np.median(cc[c])))
        out[k] = {"cand": ci, "hom_med": float(np.median(cc[ci])), "hom_seeds": cc[ci]}
    return out


def sensor_model(plant):
    """The plant's model compiled again from its XML with one contact sensor per finger (net force between the tip body,
    which carries the pads, and the cylinder), the palm placed as on the plant."""
    import xml.etree.ElementTree as ET
    mj = plant.mj
    path = Path(plant.xml_path)
    root = ET.parse(path).getroot()
    sens = root.find("sensor")
    if sens is None:
        sens = ET.SubElement(root, "sensor")
    for f in S.FINGERS:
        ET.SubElement(sens, "contact", name=f"mpc_{f}", body1=f"{f}_tip", body2=S.OBJ, data="force", reduce="netforce",
                      num="1")
    tmp = path.with_name(f"{path.stem}.mpcsens.{os.getpid()}.xml")
    ET.ElementTree(root).write(tmp)
    try:
        m = mj.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()
    m.body_pos[m.body("palm_pose").id] = plant.m.body_pos[plant.m.body("palm_pose").id]
    if m.nq != plant.m.nq or m.nv != plant.m.nv:
        raise RuntimeError("sensor model changed the state size")
    return m


class Planner:
    """Predictive sampling over a piecewise-linear spline of joint targets, rolled out on the plant's model."""

    def __init__(self, plant, prm, nthread: int, rng):
        from mujoco import rollout
        import copy
        mj = plant.mj
        self.mj, self.plant, self.prm, self.rng = mj, plant, prm, rng
        # the planner's model has no post, so no plan leans on it (2026-10-13: with the post in its model the first
        # test pivoted the tool 48 deg on the post in 0.25 s and dropped it when the post went away at 1.3 s)
        if prm.get("w_contact", 0.0) > 0:
            self.m = m = sensor_model(plant)
        else:
            self.m = m = copy.deepcopy(plant.m)
        post = m.geom("post").id
        m.geom_contype[post] = m.geom_conaffinity[post] = 0
        if prm.get("pad_filter", 0.0) > 0:
            # pads collide with the cylinder only (the RL scenes' rule): the scale-study scenes let every pad meet every
            # geom, 3.4 million allowed pairs, which the planner's model need not check
            pads = np.array([g for g in range(m.ngeom) if "_pad" in (m.geom(g).name or "")])
            tool = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] == plant.tb])
            m.geom_contype[pads], m.geom_conaffinity[pads] = 2, 0
            m.geom_conaffinity[tool] = m.geom_conaffinity[tool] | 2
        self.pool = rollout.Rollout(nthread=nthread)
        self.datas = [mj.MjData(m) for _ in range(nthread)]
        self.spec = mj.mjtState.mjSTATE_FULLPHYSICS
        self.x0 = np.empty(mj.mj_stateSize(m, self.spec))
        self.dt = float(m.opt.timestep)
        self.nh = int(round(prm["horizon"] / self.dt))
        self.knot_t = np.linspace(0.0, prm["horizon"], int(prm["knots"]))
        tk = np.arange(self.nh) * self.dt
        self.W = np.stack([np.interp(tk, self.knot_t, np.eye(len(self.knot_t))[k]) for k in range(len(self.knot_t))], 1)
        self.lo, self.hi = m.actuator_ctrlrange[plant.act, 0], m.actuator_ctrlrange[plant.act, 1]
        self.qa = 1 + int(plant.qa)            # FULLPHYSICS = [time, qpos, qvel, act, ...]
        self.wa = 1 + m.nq + int(plant.va) + 3   # the free joint's angular velocity in the state vector
        self.nominal = None
        self.t_nom = None
        self.ctrl_full = np.zeros((int(prm["nsample"]), self.nh, m.nu))
        self.ctrl_full[:] = plant.d.ctrl

    def shift(self, t):
        """The nominal spline evaluated at the new knot times (held at its last knot beyond its end)."""
        old_t = self.t_nom + self.knot_t
        new_t = t + self.knot_t
        self.nominal = np.stack([np.interp(new_t, old_t, self.nominal[:, j]) for j in range(self.nominal.shape[1])], 1)
        self.t_nom = t

    def cost(self, states, samples, p_ref, z0, sens=None):
        prm = self.prm
        q = states[:, prm["cost_every"] - 1::prm["cost_every"], self.qa:self.qa + 7]
        p, x, y = q[..., :3], q[..., 4], q[..., 5]
        uz = 1.0 - 2.0 * (x * x + y * y)
        run = prm["w_tilt"] * (1.0 - uz) + prm["w_pos"] * np.sum((p - p_ref) ** 2, -1) / SIGMA_POS ** 2 \
            + prm["w_drop"] * (p[..., 2] < z0 - 0.015)
        if prm.get("w_rate", 0.0) > 0:
            om = np.linalg.norm(states[:, prm["cost_every"] - 1::prm["cost_every"], self.wa:self.wa + 3], axis=-1)
            wm = math.radians(prm["rate_max_deg"])
            run = run + prm["w_rate"] * np.maximum(0.0, om - wm) ** 2 / wm ** 2
        if prm.get("w_contact", 0.0) > 0 and sens is not None:
            F = np.linalg.norm(sens[:, prm["cost_every"] - 1::prm["cost_every"], :].reshape(
                sens.shape[0], -1, len(S.FINGERS), 3), axis=-1)
            run = run + prm["w_contact"] * np.clip((prm["f_min"] - F) / prm["f_min"], 0.0, 1.0).sum(-1)
        c = run[:, :-1].sum(1) + prm["w_term"] * run[:, -1]
        return c + prm["w_du"] * np.sum(np.diff(samples, axis=1) ** 2, (1, 2))

    def evaluate(self, samples, p_ref, z0):
        """Roll out knot sets (n, K, 9) from the stored state; their costs and controls."""
        m = self.m
        ctrl = np.einsum("hk,nkj->nhj", self.W, samples)
        full = np.repeat(self.ctrl_full[:1], len(samples), axis=0)
        full[:, :, self.plant.act] = ctrl
        states, sens = self.pool.rollout(m, self.datas, self.x0[None], full, nstep=self.nh)
        c = self.cost(states, samples, p_ref, z0, sens if m.nsensordata else None)
        return c, ctrl

    def noise(self, n, scale):
        """Gaussian knot noise (n, K, 9) times `scale` (K, 1) or scalar; knot-correlated AR(1) for iCEM."""
        K = len(self.knot_t)
        e = self.rng.normal(0.0, 1.0, (n, K, self.nominal.shape[1]))
        if self.prm.get("method") == "icem" and self.prm["ar_alpha"] > 0:
            a = self.prm["ar_alpha"]
            for h in range(1, K):
                e[:, h] = a * e[:, h - 1] + math.sqrt(1.0 - a * a) * e[:, h]
        return e * scale

    def constrain(self, samples, u_now):
        samples = np.clip(samples, self.lo, self.hi)
        if self.prm.get("pin0", 0.0) > 0:
            samples[:, 0] = u_now
        return samples

    def mppi_update(self, samples, c):
        lam = self.prm["lam_rel"] * (float(np.std(c)) + 1e-9)
        w = np.exp(-(c - c.min()) / lam)
        w /= w.sum()
        return np.einsum("n,nkj->kj", w, samples)

    def plan(self, t, p_ref, z0):
        prm, d = self.prm, self.plant.d
        u_now = d.ctrl[self.plant.act].copy()
        if self.nominal is None:
            self.nominal = np.tile(u_now, (len(self.knot_t), 1))
            self.t_nom = t
            self.elites = None
        else:
            if getattr(self, "elites", None) is not None:      # iCEM: shift the kept elites with the nominal
                old_t, new_t = self.t_nom + self.knot_t, t + self.knot_t
                self.elites = np.stack([np.stack([np.interp(new_t, old_t, e[:, j]) for j in range(e.shape[1])], 1)
                                        for e in self.elites])
            self.shift(t)
        self.mj.mj_getState(self.plant.m, d, self.x0, self.spec)
        meth = prm.get("method", "ps")
        B, I = int(prm["nsample"]), max(1, int(prm.get("iters", 1)))
        K = len(self.knot_t)
        sig = prm["sigma"]
        if meth == "ps":
            samples = self.constrain(self.nominal[None] + self.noise(B, sig), u_now)
            samples[0] = self.constrain(self.nominal[None], u_now)[0]
            c, ctrl = self.evaluate(samples, p_ref, z0)
            b = int(np.argmin(c))
            self.nominal = samples[b]
            return ctrl[b], float(c[b]), float(c[0])
        n = max(2, B // I)
        if meth in ("mppi", "anneal"):
            h = np.arange(K)[:, None]
            best = (np.inf, None)
            for i in range(I):
                scale = sig if meth == "mppi" else sig * np.exp(-i / (prm["beta1"] * I) - (K - 1 - h) / (prm["beta2"] * K))
                samples = self.constrain(self.nominal[None] + self.noise(n, scale), u_now)
                samples[0] = self.constrain(self.nominal[None], u_now)[0]
                c, ctrl = self.evaluate(samples, p_ref, z0)
                if c.min() < best[0]:
                    best = (float(c.min()), None)
                self.nominal = self.constrain(self.mppi_update(samples, c)[None], u_now)[0]
            ctrl_exec = np.einsum("hk,kj->hj", self.W, self.nominal)
            return ctrl_exec, best[0], float(c[0])
        if meth in ("cem", "icem"):
            mean = self.nominal.copy()
            std = np.full_like(mean, sig)
            ne = max(2, int(round(prm["elite_frac"] * n)))
            best = (np.inf, None, None)
            for i in range(I):
                samples = self.constrain(mean[None] + self.noise(n, std), u_now)
                samples[0] = self.constrain(mean[None], u_now)[0]
                if meth == "icem" and i == 0 and getattr(self, "elites", None) is not None:
                    kk = min(len(self.elites), n - 1)
                    samples[n - kk:] = self.constrain(self.elites[:kk], u_now)
                c, ctrl = self.evaluate(samples, p_ref, z0)
                idx = np.argsort(c)
                if c[idx[0]] < best[0]:
                    best = (float(c[idx[0]]), samples[idx[0]].copy(), ctrl[idx[0]].copy())
                el = samples[idx[:ne]]
                mom = prm["momentum"]
                mean = (1 - mom) * el.mean(0) + mom * mean
                std = np.clip((1 - mom) * el.std(0) + mom * std, prm["sigma_min"], sig)
                if meth == "icem":
                    self.elites = samples[idx[:int(prm["keep"])]].copy()
            if meth == "icem":
                self.nominal = best[1]
                return best[2], best[0], float(c[0])
            self.nominal = self.constrain(mean[None], u_now)[0]
            return np.einsum("hk,kj->hj", self.W, self.nominal), best[0], float(c[0])
        raise ValueError(f"unknown method {meth}")


def mpc_turn(plant, meta, seed, prm, nthread=8, frame_cb=None) -> dict:
    """One MPC turn on a cell's candidate grasp; the row has the fields of hand_object_scale_sim.turn_rollout's."""
    mj, m, d = plant.mj, plant.m, plant.d
    post = m.geom("post").id
    if not hasattr(plant, "post0"):
        plant.post0 = (int(m.geom_contype[post]), int(m.geom_conaffinity[post]))
    m.geom_contype[post], m.geom_conaffinity[post] = plant.post0
    grip = np.array([meta["grip"][f][k] for f in S.FINGERS for k in range(3)])
    plant.reset(seed)
    plant.set_targets(grip)
    if frame_cb is None:
        plant.advance(prm["t_grip"])
    else:
        tg = 0.0
        while tg < prm["t_grip"] - 1e-9:
            tg = min(prm["t_grip"], tg + 0.01)
            plant.advance(tg)
            frame_cb(plant, tg, {"phase": "grip", "th_deg": 0.0})
    s0 = plant.state()
    u0 = s0["R"][:, 2] / np.linalg.norm(s0["R"][:, 2])
    a0 = np.cross(u0, [0.0, 0.0, 1.0])
    a0 /= np.linalg.norm(a0)
    h0 = np.cross(a0, u0)
    theta = lambda R: math.atan2(R[:, 2] @ h0, R[:, 2] @ u0)  # noqa: E731  (hom_turn3.Turn3.theta)
    z0 = float(s0["p"][2])
    p_ref = s0["p"] + [0.0, 0.0, prm["lift_mm"] * 1e-3]
    weight = meta["mass_kg"] * 9.81
    rng = np.random.default_rng([int(seed), 20261013])
    if prm.get("method") == "ilqr":
        from hand_object_scale_ilqr import ILQRPlanner
        pl = ILQRPlanner(plant, prm, nthread, rng)
    elif prm.get("device", "cpu") == "gpu":
        from hand_object_scale_mpc_gpu import GPUPlanner
        pl = GPUPlanner(plant, prm, nthread, rng, nworld=max(2, int(prm["nsample"]) // max(1, int(prm.get("iters", 1)))))
    else:
        pl = Planner(plant, prm, nthread, rng)
    t_sq_end = prm["t_grip"] + prm["t_squeeze"]
    t_post = prm.get("t_post_off", t_sq_end)
    n_exec = int(round(prm["dt_plan"] / pl.dt))
    th_max = th2_max = th3_max = -10.0
    first_loss = None
    trace = []
    log, n_step, log_every = [], 0, int(prm.get("log_every", 10))
    w0 = time.perf_counter()
    t_plan = 0.0
    while d.time < prm["t_end"] - 1e-9:
        if d.time >= t_post - 1e-9 and m.geom_contype[post]:
            m.geom_contype[post] = m.geom_conaffinity[post] = 0
        s = plant.state()
        th = theta(s["R"])
        nf = sum(1 for f in S.FINGERS if s["n"][f] > 0 and s["F"][f] >= weight)
        if d.time >= t_sq_end and nf < 3 and first_loss is None:
            first_loss = round(math.degrees(th), 2)
        if nf >= 2 and s["p"][2] > z0 - 0.020:
            th2_max = max(th2_max, th)
            if nf == 3:
                th3_max = max(th3_max, th)
        th_max = max(th_max, th)
        tp = time.perf_counter()
        u, c_best, c_nom = pl.plan(float(d.time), p_ref, z0)
        t_plan += time.perf_counter() - tp
        if not trace or d.time - trace[-1]["t"] >= 0.05 - 1e-9:
            trace.append({"t": round(float(d.time), 3), "th": round(math.degrees(th), 2),
                          "F": {f: round(s["F"][f], 2) for f in S.FINGERS}, "cost": round(c_best, 3)})
        for k in range(n_exec):
            d.ctrl[plant.act] = u[k]
            mj.mj_step(m, d)
            n_step += 1
            if n_step % log_every == 0:
                log.append(log_sample(plant))
        if frame_cb is not None:
            frame_cb(plant, float(d.time), {"phase": "squeeze" if d.time < t_sq_end else "turn",
                                            "th_deg": math.degrees(theta(d.xmat[plant.tb].reshape(3, 3)))})
        if d.qpos[plant.qa + 2] < z0 - 0.03 or not np.all(np.isfinite(d.qpos)):
            break
    se = plant.state()
    loaded = sum(1 for f in S.FINGERS if se["n"][f] > 0 and se["F"][f] >= weight)
    dropped = bool(se["p"][2] < z0 - 0.020)
    th_end = theta(se["R"])
    held = (not dropped) and loaded >= 2
    return {"turn_end_deg": round(math.degrees(th_end), 2), "held": held,
            "held_turn_deg": round(math.degrees(th_end), 2) if held else 0.0, "turn_max_deg": round(math.degrees(th_max), 2),
            "turn_held2_max_deg": round(math.degrees(th2_max), 2), "turn_held3_max_deg": round(math.degrees(th3_max), 2),
            "first_loss_deg": first_loss, "fingers_loaded_end": loaded, "dropped": dropped,
            "tilt_end_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, float(se["R"][2, 2]))))), 2),
            "F_end": {f: round(se["F"][f], 3) for f in S.FINGERS}, "z_drop_mm": round(1e3 * float(z0 - se["p"][2]), 2),
            "t_sim_end": round(float(d.time), 3), "wall_s": round(time.perf_counter() - w0, 2),
            "plan_s": round(t_plan, 2), "trace": trace, **smooth_metrics(log, weight, theta),
            "log": compact_log(log, theta)}


def log_sample(plant):
    """One 10 ms sample for the smoothness metrics: time, servo command, cylinder rotation and angular velocity, and
    each finger's normal force on the cylinder."""
    d = plant.d
    s = plant.state()
    return (float(d.time), d.ctrl[plant.act].copy(), d.qvel[plant.va + 3:plant.va + 6].copy(), s["R"],
            np.array([s["F"][f] for f in S.FINGERS]))


def smooth_metrics(log, weight, theta, t_from=0.8):
    """Smoothness and contact continuity of one run from its 10 ms samples after `t_from`:
      cmd_rate_rms     RMS over joints and samples of the servo-command rate, rad/s
      cmd_reversals    sign reversals of the command rate per joint per second (changes under 1e-4 rad ignored)
      omega_jitter     RMS of the cylinder's angular velocity minus its 100 ms running mean, deg/s
      peak_rate        largest |d theta / dt| over 50 ms, deg/s
      three_frac       share of samples with all three fingers pressing with at least the cylinder's weight
      contact_breaks   times a finger's force fell below the weight for at least 20 ms after holding it"""
    L = [x for x in log if x[0] >= t_from - 1e-9]
    if len(L) < 12:
        return {}
    t = np.array([x[0] for x in L])
    u = np.array([x[1] for x in L])
    w = np.array([x[2] for x in L])
    F = np.array([x[4] for x in L])
    dt = float(np.median(np.diff(t)))
    du = np.diff(u, axis=0)
    rate = du / dt
    sig = np.where(np.abs(du) > 1e-4, np.sign(du), 0.0)
    rev = 0
    for j in range(u.shape[1]):
        s_ = sig[:, j][sig[:, j] != 0]
        rev += int(np.count_nonzero(s_[1:] != s_[:-1]))
    span = t[-1] - t[0]
    k = max(1, int(round(0.1 / dt)))
    kern = np.ones(k) / k
    w_hp = np.stack([w[:, i] - np.convolve(w[:, i], kern, mode="same") for i in range(3)], 1)
    th = np.degrees([theta(x[3]) for x in L])
    m5 = max(1, int(round(0.05 / dt)))
    peak = float(np.max(np.abs(th[m5:] - th[:-m5]))) / (m5 * dt) if len(th) > m5 else 0.0
    on = F >= weight
    breaks = 0
    for i in range(F.shape[1]):
        held_ = False
        low = 0
        for v in on[:, i]:
            if v:
                held_, low = True, 0
            elif held_:
                low += 1
                if low == 2:
                    breaks += 1
                    held_ = False
    return {"cmd_rate_rms": round(float(np.sqrt(np.mean(rate ** 2))), 4),
            "cmd_reversals": round(rev / (u.shape[1] * span), 3),
            "omega_jitter": round(float(np.degrees(np.sqrt(np.mean(w_hp[k:-k] ** 2)))), 3) if len(w) > 2 * k else None,
            "peak_rate": round(peak, 1), "three_frac": round(float(np.mean(on.all(1))), 4), "contact_breaks": breaks}


def compact_log(log, theta, every=2):
    """The 10 ms samples thinned to 20 ms for the rows: t, turn (deg), finger forces (N), |omega| (deg/s), command."""
    return [{"t": round(x[0], 3), "th": round(math.degrees(theta(x[3])), 2), "F": [round(float(v), 2) for v in x[4]],
             "w": round(float(np.degrees(np.linalg.norm(x[2]))), 1), "u": [round(float(v), 4) for v in x[1]]}
            for x in log[::every]]


def hom_turn_logged(plant, mirror, meta, seed, hom_prm, frame_cb=None):
    """hand_object_scale_sim.turn_rollout with the 10 ms log and the smoothness metrics of an MPC run."""
    log = []

    def cb(pl, t, info):
        log.append(log_sample(pl))
        if frame_cb is not None:
            frame_cb(pl, t, info)

    res = S.turn_rollout(plant, mirror, meta, seed, hom_prm, frame_cb=cb)
    s0_R = next(x[3] for x in log if x[0] >= hom_prm["t_grip"] - 1e-9)
    u0 = s0_R[:, 2] / np.linalg.norm(s0_R[:, 2])
    a0 = np.cross(u0, [0.0, 0.0, 1.0])
    a0 /= np.linalg.norm(a0)
    h0 = np.cross(a0, u0)
    theta = lambda R: math.atan2(R[:, 2] @ h0, R[:, 2] @ u0)  # noqa: E731
    weight = meta["mass_kg"] * 9.81
    res.update(smooth_metrics(log, weight, theta))
    res["log"] = compact_log(log, theta)
    return res


def setup(cell, shape, dd, ci):
    """The plant of one cell's candidate grasp, as hand_object_scale_sim.cmd_run builds it."""
    import mujoco
    kp = S.RB.plant_args(S.PLANT)["kp"]
    cand = S.candidates(cell, shape, dd)[ci]
    sc = S.cell_scenes(cell, shape, dd, models=("pads", "pt"))
    m_pt = mujoco.MjModel.from_xml_path(str(sc["pt"][0]))
    gg = S.grasp_geometry(cell, shape, dd, cand["h"], cand["spread"])
    gp = S.grip_poses(m_pt, gg, kp)
    path, meta0 = sc["pads"]
    meta = S.candidate_meta(meta0, gg, gp)
    plant = H.MjPlant(path, meta)
    plant.xml_path = str(path)
    S.set_palm(plant.m, gg["palm"])
    geo = {"cand": ci, "h_mm": round(cand["h"] * 1000, 2), "spread_mm": round(cand["spread"] * 1000, 2)}
    return plant, meta, geo, sc, gg


def cmd_run(a) -> int:
    if a.plant:
        S.PLANT = a.plant
    prm = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=")
        prm[k] = float(v)
    for k in ("nsample", "knots", "cost_every"):
        prm[k] = int(prm[k])
    out_dir = Path(a.out_dir) if a.out_dir else OUT / "mpc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "turn.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok":
                done.add((r["tag"], r["shape"], r["d_mm"], r["cand"], r["seed"]))
    best = hom_best(Path(a.hom_rows))
    todo = [c for c in S.cell_list(a.cells) if c[1] in a.shapes.split(",") and c[0]["feasible"]]
    if a.only:
        keep = []
        for spec in a.only:
            f = spec.split(":")
            ts = {float(x) for x in f[2].split(",")} if len(f) > 2 else None
            keep += [c for c in todo if c[1] == f[0] and c[2] == float(f[1]) and (ts is None or c[0].get("t") in ts)]
        todo = keep
    k0, n0 = (int(v) for v in a.shard.split("/"))
    todo = [c for i, c in enumerate(todo) if i % n0 == k0]
    print(f"{len(todo)} cells in shard {a.shard}; prm {prm}", flush=True)
    for cell, shape, dd in todo:
        key = (cell["tag"], shape, dd)
        if key not in best:
            print(f"skip {key}: no HOM rows", flush=True)
            continue
        ci = best[key]["cand"]
        seeds = [s_ for s_ in a.seeds if (cell["tag"], shape, dd, ci, s_) not in done]
        if not seeds:
            continue
        base = {"hand_model": S.HAND, "tag": cell["tag"], "t": cell.get("t"), "x_sep_mm": cell["x_sep_mm"], "y_sep_mm": cell["y_sep_mm"],
                "shape": shape, "d_mm": dd, "model": "pads", "plant": S.PLANT, "controller": "mpc",
                "hom_med_deg": best[key]["hom_med"],
                "hom_seeds_deg": best[key]["hom_seeds"], **{f"prm_{k}": v for k, v in prm.items()}}
        try:
            plant, meta, geo, _, _ = setup(cell, shape, dd, ci)
        except Exception as e:
            print(f"ERROR setup {key}: {e}", flush=True)
            continue
        for s_ in seeds:
            row = {**base, **geo, "seed": s_, "when": time.strftime("%Y-%m-%d %H:%M")}
            try:
                res = mpc_turn(plant, meta, s_, prm, nthread=a.nthread)
                if not a.traces:
                    res.pop("trace", None)
                row.update(res, status="ok")
            except Exception as e:
                row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1200:])
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            print(f"mpc {cell['tag']} {shape} {dd:5.1f} c{ci} s{s_}: " + (
                f"held {row['held']} turn {row['turn_end_deg']} (HOM {best[key]['hom_med']:.1f}) loaded "
                f"{row['fingers_loaded_end']} F {row['F_end']} {row['wall_s']} s (plan {row['plan_s']} s)"
                if row["status"] == "ok" else row["error"]), flush=True)
    return 0


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def big_label(frame, lines, size=30):
    """Label lines in a 30 px font, so that a 640 px panel shown a third of an 828 px column wide keeps 13 px text."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im)
    font = ImageFont.truetype(FONT, size)
    y = 6
    for ln in lines:
        x0, y0, x1, y1 = dr.textbbox((10, y), ln, font=font)
        dr.rectangle([4, y - 2, x1 + 6, y1 + 4], fill=(255, 255, 255))
        dr.text((10, y), ln, fill=(15, 15, 15), font=font)
        y = y1 + 8
    return np.asarray(im)


def film_pair(cell, shape, dd, ci, seed, prm, hom_prm, nthread, controllers=("HOM", "MPC"), prm3=None, planners=None):
    """The HOM turn and the MPC turns of one cell's candidate grasp, rendered at 25 frames per simulated second with
    the pads touching the object drawn red (hand_object_scale_films' camera and labels). `planners` maps further
    controller names to (panel title, parameters) for mpc_turn (hand_object_scale_planners.py film)."""
    planners = planners or {}
    import mujoco
    import hand_object_scale_films as FL
    import reorient_backends_films as F
    out = {}
    for ctrl in controllers:
        plant, meta, geo, sc, gg = setup(cell, shape, dd, ci)
        fpath = FL.film_scene(sc["pads"][0], dd / 2000.0)
        plant = H.MjPlant(fpath, meta)
        plant.xml_path = str(fpath)
        S.set_palm(plant.m, gg["palm"])
        m, d = plant.m, plant.d
        pads = [i for i in range(m.ngeom) if "_pad" in (m.geom(i).name or "")]
        for i in pads:
            m.geom_rgba[i] = (0.75, 0.4, 0.25, 0.0)
        rend, cam = F.renderer_for(m, np.asarray(meta["tool7"], float)[:3])
        cam.distance = 0.8 * (0.24 + 1.4 * dd / 1000.0)
        if S.HAND == "cad_v1":   # the printed links hide the cylinder from real_v1's camera; look up from 10 deg below the grasp
            cam.elevation = 10.0
        frames, k = [], [0]
        every = 4 if ctrl == "HOM" else max(1, int(round(0.04 / prm["dt_plan"])))
        title = {"HOM": "HOM", "MPC": "MPC", "MPC3": "MPC, three pads"}.get(ctrl) or planners[ctrl][0]

        post_id = m.geom("post").id

        def cb(pl, t, info):
            k[0] += 1
            if k[0] % (4 if info.get("phase") == "grip" else every):
                return
            m.geom_rgba[post_id, 3] = 1.0 if m.geom_contype[post_id] else 0.0   # drawn only while it collides
            m.geom_rgba[pads, 3] = 0.0
            for i in range(d.ncon):
                g1, g2 = int(d.contact[i].geom[0]), int(d.contact[i].geom[1])
                for g, o in ((g1, g2), (g2, g1)):
                    if o in pl.tool_geoms and m.geom_bodyid[g] != pl.tb and "_pad" in (m.geom(g).name or ""):
                        m.geom_rgba[g] = (0.9, 0.1, 0.1, 1.0)
            rend.update_scene(d, cam)
            frames.append(big_label(rend.render().copy(), [title, f"{t:4.2f} s   {info.get('th_deg', 0):+5.1f}\u00b0"]))

        if ctrl == "HOM":
            mirror = S.Mirror(Path(sc["pt"][0]))
            S.set_palm(mirror.m, gg["palm"])
            res = S.turn_rollout(plant, mirror, meta, seed, hom_prm, frame_cb=cb)
        elif ctrl in planners:
            res = mpc_turn(plant, meta, seed, planners[ctrl][1], nthread=nthread, frame_cb=cb)
        elif ctrl == "MPC":
            res = mpc_turn(plant, meta, seed, prm, nthread=nthread, frame_cb=cb)
        else:
            res = mpc_turn(plant, meta, seed, prm3, nthread=nthread, frame_cb=cb)
        while len(frames) < int(25 * max(prm["t_end"], (prm3 or prm)["t_end"])):
            frames.append(frames[-1])
        rend.close()
        out[ctrl] = (frames, res, geo)
    return out


def cmd_film(a) -> int:
    import imageio
    import reorient_backends_films as F
    if a.plant:
        S.PLANT = a.plant
    F.W, F.H = (int(v) for v in a.size.split("x"))   # frame size of every panel (the shared helpers read these)
    prm = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=")
        prm[k] = float(v)
    for k in ("nsample", "knots", "cost_every"):
        prm[k] = int(prm[k])
    hom_prm = dict(H.DEFAULTS, post_off=1.0, kp_model=S.RB.plant_args(S.PLANT)["kp"])
    prm3 = dict(prm, **CONTROLLED)
    for kv in a.set3:
        k, v = kv.split("=")
        prm3[k] = float(v)
    for k in ("nsample", "knots", "cost_every"):
        prm3[k] = int(prm3[k])
    best = hom_best(Path(a.hom_rows))
    media = Path(a.out_dir) if a.out_dir else OUT / "media"
    media.mkdir(parents=True, exist_ok=True)
    for spec in a.cells:
        shape, dd, t = spec.split(":")
        dd, t = float(dd), float(t)
        cell, _, _ = next(c for c in S.cell_list("diag") if c[1] == shape and c[2] == dd and c[0].get("t") == t)
        ci = best[(cell["tag"], shape, dd)]["cand"]
        pair = film_pair(cell, shape, dd, ci, a.seed, prm, hom_prm, a.nthread, tuple(a.controllers.split(",")), prm3)
        clips = []
        stem = f"{'_'.join(c.lower() for c in pair)}_{S.obj_tag(shape, dd)}_{cell['tag']}_s{a.seed}"
        tpath = media / "films_traces.json"
        traces = json.loads(tpath.read_text()) if tpath.exists() else {}
        traces[stem] = {"shape": shape, "d_mm": dd, "t": t, "tag": cell["tag"], "x_sep_mm": cell["x_sep_mm"],
                        "y_sep_mm": cell["y_sep_mm"], "seed": a.seed, "cand": ci,
                        **{c: r.get("trace", []) for c, (_, r, _) in pair.items()}}
        tpath.write_text(json.dumps(traces))
        for ctrl, (frames, res, geo) in pair.items():
            res = {k: v for k, v in res.items() if k != "trace"}
            clip = media / f"{ctrl.lower()}_{S.obj_tag(shape, dd)}_{cell['tag']}_s{a.seed}.mp4"
            imageio.mimwrite(str(clip), frames, fps=25, codec="libx264", quality=6, macro_block_size=8)
            clips.append(clip)
            row = {"clip": clip.name, "controller": ctrl, "hand_model": S.HAND, "plant": S.PLANT, "tag": cell["tag"], "t": t, "shape": shape,
                   "d_mm": dd, "seed": a.seed, **geo, "held": res["held"], "turn_end_deg": res["turn_end_deg"],
                   "held_turn_deg": res["held_turn_deg"], "fingers_loaded_end": res["fingers_loaded_end"],
                   "F_end": res["F_end"], "frames": len(frames)}
            with open(media / "films.jsonl", "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        n = F.tile_stream(clips, media / f"{stem}.mp4", media / f"{stem}.jpg", cols=len(clips), crf=a.crf)
        print(f"tiled {n} frames -> {stem}.mp4", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cells", default="diag", help="coarse | diag | xsep | ysep (hand_object_scale_sim.cell_list)")
    r.add_argument("--shapes", default="cylinder,sphere")
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="object placement jitter, as HOM's")
    r.add_argument("--set", nargs="*", default=[], help=f"controller parameters k=v; defaults {DEFAULTS}")
    r.add_argument("--nthread", type=int, default=8, help="rollout threads")
    r.add_argument("--shard", default="0/1")
    r.add_argument("--only", nargs="*", default=[], help="cells shape:d_mm[:t,t,...]")
    r.add_argument("--plant", default=None, help=f"plant spec (reorient_backends.plant_args); default {S.PLANT}")
    r.add_argument("--hom-rows", default=str(HOM_ROWS), help="HOM turn rows that choose each cell's grasp")
    r.add_argument("--out-dir", default=None)
    r.add_argument("--traces", action="store_true")
    f = sub.add_parser("film")
    f.add_argument("--cells", nargs="+", required=True, help="shape:d_mm:t, e.g. cylinder:25:0.125")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--plant", default=None)
    f.add_argument("--set", nargs="*", default=[])
    f.add_argument("--nthread", type=int, default=6)
    f.add_argument("--hom-rows", default=str(HOM_ROWS))
    f.add_argument("--controllers", default="HOM,MPC,MPC3", help="panels: HOM, MPC (swing allowed), MPC3 (controlled)")
    f.add_argument("--set3", nargs="*", default=[], help="parameters of the controlled variant beyond CONTROLLED")
    f.add_argument("--size", default="640x480", help="panel width x height in pixels")
    f.add_argument("--crf", type=int, default=22, help="x264 quality of the tiled film (lower is better)")
    f.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    return {"run": cmd_run, "film": cmd_film}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
