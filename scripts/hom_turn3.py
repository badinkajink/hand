#!/usr/bin/env python3
"""The three-finger reorientation of the bench, driven by the relative contact-velocity controller of arXiv
2609.25619 (Wang, Oh, Pollard) instead of the deployed open-loop joint trajectories.

Scene: each deployed hand D1-D8 on its bench scene (reorient_backends.build_scene: palm welded at the plan's pose,
the 25 mm tool lying on the post, the printed TPU tip with 2.7 mm fillets as 1 mm sphere pads in MuJoCo or as a
compliant convex in Drake). The fingers first take the plan's grip pose (0.8 s), as the bench does.

Controller (100 Hz). The paper's generalized contact frame (GCF) per fingertip: origin at the witness points of the
TPU block (convex mesh, `mj_geomDistance` on a kinematic MuJoCo copy, the Mirror) and the tool cylinder, x = the tool's
surface normal at the contact (tool -> finger), y = the finger's flexion axis projected on the tangent plane,
z = x cross y. The paper commands relative contact velocities; for a turn with all three contacts held, the reference
of each contact is the velocity of the tool's material point under the DESIRED tool twist (so the relative velocity
is zero when the tool follows the reference), plus a normal component that regulates the pad force:

    tool twist   w_d = a0 (dtheta_ref + K_R (theta_ref - theta)) + K_R (u x u_ref),   v_d = K_P (c_ref - c)
    contact i    v_i = v_d + w_d x (p_i - c) + n_i K_F (F_i - F_des)        (in the GCF: sep, slide_y, slide_z)
                 w_i = w_d                                                   (roll/spin rows, weight W_ROT)

with theta the tool's tilt from horizontal (asin of its axis' z component), a0 the horizontal axis normal to the
tool, u_ref the tool axis rotated by theta_ref about a0, c the tool's centre. Eq. 2 of the paper (bounded least
squares per finger, joint-rate damping, posture term, joint limits) gives joint rates u; the servo targets integrate
u and stay within LEAD of the measured angles. The force F_i is the plant's own normal force on finger i's pads.

Phases: grip 0.8 s (plan pose) | squeeze 0.5 s (theta_ref = theta0, the grip force relaxes to F_des) | turn at
RATE to GOAL | hold 2 s. The governor (on by default) stops the reference where a pad's force falls below 30 % of F_des
or the tool lags it by 6 deg, and holds that angle.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/hom_turn3.py run --hands D7 --sims mujoco --seeds 0
Rows: docs/experiments/20261006-hom_turn3/turn3.jsonl, one fsynced line per rollout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reorient_backends as RB  # noqa: E402

OUT = ROOT / "docs/experiments/20261006-hom_turn3/turn3.jsonl"
SCENES = ROOT / "logs/20261006-hom_turn3/scenes"
FINGERS, JOINTS = RB.FINGERS, RB.JOINTS
NAMES = [f"{f}_{j}" for f in FINGERS for j in JOINTS]
ELL = 0.02                    # m; linear rows scaled by 1/ELL (the paper's weighting of 20 mm/s against 1 rad/s)
W_LIN, W_ROT, W_DAMP, W_POST = 1.0, 0.01, 1e-3, 1e-5
U_MAX, LEAD = 2.0, 0.3
DEFAULTS = dict(rate_deg=30.0, goal_deg=90.0, f_des=2.0, k_f=0.03, k_r=3.0, k_p=3.0, lift_mm=5.0, dt_c=0.01,
                t_grip=0.8, t_squeeze=0.5, t_hold=2.0, v_approach=0.02, centre=1.0, governor=1.0, lim_deg=3.0, f_frac=0.3,
                err_deg=6.0, jt=0.0, kp_model=4.0, tau_servo=0.02, k_fi=2.0, gov_contact=0.0)
PLANT = "kp4_kv0_fr1_fl0_dp0.08"   # 2026-10-06 bench-replay fit (coarse): kp 4 N m/rad, tau 0.02 s, 1 N m


# ---------------------------------------------------------------------------------- scenes

def scenes(hand, plant, mu):
    """(pads scene, meta) for the plant and (mesh scene) for the controller's Mirror; same hand, plant, friction."""
    p_pads, meta = RB.build_scene(hand, "tpu2.7", "pads", plant, "bed", 100.0, mu, out_dir=SCENES)
    p_mesh, meta_mesh = RB.build_scene(hand, "tpu2.7", "pt", plant, "bed", 100.0, mu, out_dir=SCENES)
    return p_pads, p_mesh, meta, meta_mesh


class Mirror:
    """Kinematic copy of the bench scene with the TPU block as one convex mesh per tip."""

    def __init__(self, path):
        import mujoco
        self.mj = mujoco
        self.m = m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(m)
        self.tb = m.body(RB.OBJ).id
        self.qa = m.jnt_qposadr[m.body_jntadr[self.tb]]
        self.va = m.jnt_dofadr[m.body_jntadr[self.tb]]
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in NAMES])
        self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in NAMES])
        self.lo = np.array([m.jnt_range[m.joint(n).id][0] for n in NAMES])
        self.hi = np.array([m.jnt_range[m.joint(n).id][1] for n in NAMES])
        self.cyl = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.tb
                    and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER][0]
        self.r_cyl, self.hl = float(m.geom_size[self.cyl][0]), float(m.geom_size[self.cyl][1])
        self.tipg = {f: m.geom(f"{f}_tipgeom").id for f in FINGERS}
        self.tipb = {f: m.body(f"{f}_tip").id for f in FINGERS}
        self.flex = {f: m.joint(f"{f}_pip").id for f in FINGERS}
        self.jacp, self.jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def set_state(self, q, tool_pos, tool_quat):
        d = self.d
        d.qpos[self.qadr] = q
        d.qpos[self.qa:self.qa + 3] = tool_pos
        d.qpos[self.qa + 3:self.qa + 7] = tool_quat
        self.mj.mj_kinematics(self.m, d)
        self.mj.mj_comPos(self.m, d)

    def gcf(self, f):
        """Witness points, tool normal and frame between finger f's TPU block and the tool cylinder."""
        m, d = self.m, self.d
        ft = np.zeros(6)
        dist = self.mj.mj_geomDistance(m, d, self.tipg[f], self.cyl, 0.05, ft)
        p_f, p_o = ft[:3].copy(), ft[3:].copy()
        c = d.geom_xpos[self.cyl]
        u = d.geom_xmat[self.cyl].reshape(3, 3)[:, 2]
        s = float((p_o - c) @ u)
        rad = p_o - c - s * u
        if abs(s) < self.hl - 1e-4 and np.linalg.norm(rad) > 1e-6:
            n = rad / np.linalg.norm(rad)
        else:
            n = u * math.copysign(1.0, s)
        y = d.xaxis[self.flex[f]] - (d.xaxis[self.flex[f]] @ n) * n
        if np.linalg.norm(y) < 1e-6:
            y = np.cross(n, u)
        y /= np.linalg.norm(y)
        z = np.cross(n, y)
        return {"dist": float(dist), "p_f": p_f, "p_o": p_o, "n": n, "E": np.column_stack([n, y, z])}

    def gravity(self):
        """Joint torques holding the nine finger joints against gravity at the mirror's posture (qfrc_bias at rest)."""
        m, d = self.m, self.d
        d.qvel[:] = 0.0
        self.mj.mj_comVel(m, d)
        out = np.zeros(m.nv)
        self.mj.mj_rne(m, d, 0, out)
        return out[self.vadr]

    def jac(self, f, p):
        self.mj.mj_jac(self.m, self.d, self.jacp, self.jacr, p, self.tipb[f])
        cols = self.vadr[3 * FINGERS.index(f):3 * FINGERS.index(f) + 3]
        return self.jacp[:, cols].copy(), self.jacr[:, cols].copy()


# ---------------------------------------------------------------------------------- plants

class MjPlant:
    def __init__(self, path, meta):
        import mujoco
        self.mj = mujoco
        self.m = m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(m)
        self.meta = meta
        self.tb = m.body(RB.OBJ).id
        self.qa = m.jnt_qposadr[m.body_jntadr[self.tb]]
        self.va = m.jnt_dofadr[m.body_jntadr[self.tb]]
        self.act = np.array([m.actuator(f"a_{n}").id for n in NAMES])
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in NAMES])
        self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in NAMES])
        self.tool_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == self.tb}
        self.finger_of = {}
        for i in range(m.ngeom):
            bn = m.body(m.geom_bodyid[i]).name
            for f in FINGERS:
                if bn.startswith(f + "_"):
                    self.finger_of[i] = f
        self.f6 = np.zeros(6)

    def reset(self, seed):
        m, d = self.m, self.d
        self.mj.mj_resetData(m, d)
        tool7 = np.asarray(self.meta["tool7"], float)
        dx, dy, dyaw = RB.jitter(seed)
        d.qpos[self.qa:self.qa + 3] = tool7[:3] + [dx, dy, 0.0]
        d.qpos[self.qa + 3:self.qa + 7] = RB._yaw_quat(tool7[3:], dyaw)
        d.qpos[self.qadr] = [self.meta["q0"][n] for n in NAMES]
        self.mj.mj_forward(m, d)

    def set_targets(self, q_cmd):
        self.d.ctrl[self.act] = q_cmd

    def advance(self, t1):
        while self.d.time < t1 - 1e-9:
            self.mj.mj_step(self.m, self.d)

    def state(self):
        m, d = self.m, self.d
        F = {f: 0.0 for f in FINGERS}
        n = {f: 0 for f in FINGERS}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if (g1 in self.tool_geoms) == (g2 in self.tool_geoms):
                continue
            f = self.finger_of.get(g2 if g1 in self.tool_geoms else g1)
            if f is None:
                continue
            self.mj.mj_contactForce(m, d, i, self.f6)
            if self.f6[0] > 1e-6:
                F[f] += float(self.f6[0])
                n[f] += 1
        return {"t": float(d.time), "q": d.qpos[self.qadr].copy(), "qd": d.qvel[self.vadr].copy(),
                "p": d.qpos[self.qa:self.qa + 3].copy(), "quat": d.qpos[self.qa + 3:self.qa + 7].copy(),
                "R": d.xmat[self.tb].reshape(3, 3).copy(), "F": F, "n": n}


class DrakePlant:
    def __init__(self, path, meta):
        self.db = RB.DrakeBench(path, meta)
        self.meta = meta
        p = self.db.plant
        self.jidx = [p.GetJointByName(n) for n in NAMES]

    def reset(self, seed):
        self.db.reset(self.meta["q0"], self.meta["tool7"], seed)

    def set_targets(self, q_cmd):
        self.db.set_targets({n: float(v) for n, v in zip(NAMES, q_cmd)})

    def advance(self, t1):
        self.db.sim.AdvanceTo(t1)

    def state(self):
        db = self.db
        pc = db.pc
        q = np.array([j.get_angle(pc) for j in self.jidx])
        qd = np.array([j.get_angular_rate(pc) for j in self.jidx])
        X = db.plant.EvalBodyPoseInWorld(pc, db.tool)
        s = db.snap()
        R = X.rotation().matrix()
        wxyz = X.rotation().ToQuaternion().wxyz()
        return {"t": float(db.ctx.get_time()), "q": q, "qd": qd, "p": np.array(X.translation()),
                "quat": np.array(wxyz), "R": R, "F": s["F"], "n": s["n"]}


# ---------------------------------------------------------------------------------- controller

class Turn3:
    def __init__(self, mirror: Mirror, s0: dict, prm: dict, q0):
        self.mr, self.prm = mirror, prm
        u0 = s0["R"][:, 2]
        self.u0 = u0 / np.linalg.norm(u0)
        a0 = np.cross(self.u0, [0.0, 0.0, 1.0])
        self.a0 = a0 / np.linalg.norm(a0)
        self.h0 = np.cross(self.a0, self.u0)       # = z for a horizontal tool
        self.tilt0 = math.asin(max(-1.0, min(1.0, self.u0[2])))   # start tilt from horizontal
        self.ticks = 0
        self.limited = []
        # rotation centre: the tool's material point at the contacts' centroid (centre=1) or the tool's centre (0)
        mirror.set_state(s0["q"], s0["p"], s0["quat"])
        cen = np.mean([mirror.gcf(f)["p_o"] for f in FINGERS], axis=0) if prm["centre"] else s0["p"]
        self.c_body = s0["R"].T @ (cen - s0["p"])
        self.c_ref = cen + [0.0, 0.0, prm["lift_mm"] * 1e-3]
        self.q_cmd = np.asarray(q0, float).copy()
        self.q_post = np.asarray(q0, float).copy()
        self.sat = np.zeros(9)
        self.dF = {f: 0.0 for f in FINGERS}           # integral correction of the commanded pad force, N

    def theta(self, R):
        """Rotation of the tool axis from its start, in the vertical plane through it."""
        u = R[:, 2]
        return math.atan2(u @ self.h0, u @ self.u0)

    def step(self, s, theta_ref, dtheta_ref, f_des):
        prm, mr = self.prm, self.mr
        mr.set_state(s["q"], s["p"], s["quat"])
        u = s["R"][:, 2]
        th = self.theta(s["R"])
        u_ref = self.u0 * math.cos(theta_ref) + self.h0 * math.sin(theta_ref)
        w_d = self.a0 * (dtheta_ref + prm["k_r"] * (theta_ref - th)) + prm["k_r"] * np.cross(u, u_ref)
        c = s["p"] + s["R"] @ self.c_body
        v_d = prm["k_p"] * (self.c_ref - c)
        out = {}
        dt = prm["dt_c"]
        self.ticks += 1
        self.limited = []
        jt = prm["jt"] > 0
        tau_g = mr.gravity() if jt else None
        for k, f in enumerate(FINGERS):
            g = mr.gcf(f)
            n, E = g["n"], g["E"]
            Fi = s["F"][f]
            touching = not (s["n"][f] == 0 and g["dist"] > 0)
            if not touching:
                sep = -prm["v_approach"]
            elif jt:
                sep = 0.0
                self.dF[f] = float(np.clip(self.dF[f] + prm["k_fi"] * (f_des - Fi) * dt, -f_des, 2.0 * f_des))
            else:
                sep = prm["k_f"] * (Fi - f_des)
            v_i = v_d + np.cross(w_d, g["p_f"] - c) + n * sep
            Jp, Jr = mr.jac(f, g["p_f"])
            A = np.vstack([np.sqrt(W_LIN) * (E.T @ Jp) / ELL, np.sqrt(W_ROT) * (E.T @ Jr),
                           np.sqrt(W_DAMP) * np.eye(3), np.sqrt(W_POST) * np.eye(3)])
            sl = slice(3 * k, 3 * k + 3)
            b = np.concatenate([np.sqrt(W_LIN) * (E.T @ v_i) / ELL, np.sqrt(W_ROT) * (E.T @ w_d), np.zeros(3),
                                np.sqrt(W_POST) * (self.q_post[sl] - s["q"][sl])])
            lo = np.maximum(-U_MAX, (mr.lo[sl] - self.q_cmd[sl]) / dt)
            hi = np.minimum(U_MAX, (mr.hi[sl] - self.q_cmd[sl]) / dt)
            lo, hi = np.minimum(lo, hi - 1e-9), np.maximum(hi, lo + 1e-9)
            uf = lsq_linear(A, b, bounds=(lo, hi), method="bvls").x
            self.sat[sl] += (np.abs(uf - lo) < 1e-6) | (np.abs(uf - hi) < 1e-6)
            near = math.radians(prm["lim_deg"])
            at_lim = ((self.q_cmd[sl] - mr.lo[sl] < near) & (uf < 0)) | ((mr.hi[sl] - self.q_cmd[sl] < near) & (uf > 0))
            if np.any(at_lim):
                self.limited.append(NAMES[3 * k + int(np.argmax(at_lim))])
            if jt:
                # servo spring: target = measured + velocity lead + (J^T f + gravity) / kp  (2026-10-02 HOM chain Eq. 14)
                fc = -(f_des + self.dF[f]) * n if touching else np.zeros(3)
                tau = Jp.T @ fc + tau_g[sl]
                self.q_cmd[sl] = s["q"][sl] + uf * prm["tau_servo"] + tau / prm["kp_model"]
            else:
                self.q_cmd[sl] = self.q_cmd[sl] + uf * dt
            self.q_cmd[sl] = np.clip(self.q_cmd[sl], s["q"][sl] - LEAD, s["q"][sl] + LEAD)
            self.q_cmd[sl] = np.clip(self.q_cmd[sl], mr.lo[sl], mr.hi[sl])
            out[f] = {"dist_mm": 1e3 * g["dist"], "F": Fi, "slide_res": float(np.linalg.norm(
                (E.T @ (Jp @ uf - v_i))[1:]))}
        return self.q_cmd.copy(), th, out


# ---------------------------------------------------------------------------------- one rollout

def rollout(plant, mirror, meta, seed, prm, trace_every=0.02, frame_cb=None):
    """`frame_cb(plant, t, info)` is called every control tick (and every 10 ms of the grip) when given."""
    plan, _ = RB.load_plan(meta["hand"])
    poses = {p["name"]: p["joints"] for p in plan["poses"]}
    grip = np.array([math.radians(poses["grip"][f][j]) for f in FINGERS for j in JOINTS])
    plant.reset(seed)
    plant.set_targets(grip)
    t = 0.0
    if frame_cb is None:
        plant.advance(prm["t_grip"])
    else:
        while t < prm["t_grip"] - 1e-9:
            t = min(prm["t_grip"], t + prm["dt_c"])
            plant.advance(t)
            frame_cb(plant, t, {"phase": "grip"})
    s0 = plant.state()
    ctl = Turn3(mirror, s0, prm, grip)
    th0 = 0.0
    goal = math.radians(prm["goal_deg"]) - ctl.tilt0      # goal_deg is the tilt from horizontal (90 = vertical)
    rate = math.radians(prm["rate_deg"])
    t_sq_end = prm["t_grip"] + prm["t_squeeze"]
    t_turn_end = t_sq_end + max(0.0, goal - th0) / rate
    t_end = t_turn_end + prm["t_hold"]
    t = prm["t_grip"]
    trace = []
    th_max = th3_max = th2_max = -10.0
    w0 = time.perf_counter()
    z0 = s0["p"][2]
    frozen, n_lim, first_loss, lim_joint = None, 0, None, None
    th_cmd = th0
    while t < t_end - 1e-9:
        s = plant.state()
        nf = sum(1 for f in FINGERS if s["n"][f] > 0 and s["F"][f] > 0.05)
        th_now = ctl.theta(s["R"])
        held_now = nf >= 2 and s["p"][2] > z0 - 0.020
        if t >= t_sq_end and nf < 3 and first_loss is None:
            first_loss = (round(t, 3), round(math.degrees(th_now), 2))
        if held_now:
            th2_max = max(th2_max, th_now)
            if nf == 3:
                th3_max = max(th3_max, th_now)
        if t < t_sq_end:
            th_ref, dth, fdes = th0, 0.0, prm["f_des"]     # the plan's grip (5-13 N) relaxes to F_des
        elif frozen is not None:
            th_ref, dth, fdes = frozen, 0.0, prm["f_des"]
        else:
            th_cmd = min(goal, th_cmd + rate * prm["dt_c"])
            th_ref, dth, fdes = th_cmd, (rate if th_cmd < goal else 0.0), prm["f_des"]
        q_cmd, th, info = ctl.step(s, th_ref, dth, fdes)
        if t >= t_sq_end and frozen is None and prm["governor"]:
            # a pad is letting go, or the tool lags the reference: hold the angle reached
            if prm["gov_contact"]:
                weak = nf < 3 or th_ref - th > math.radians(prm["err_deg"])
            else:
                weak = min(s["F"].values()) < prm["f_frac"] * prm["f_des"] or th_ref - th > math.radians(prm["err_deg"])
            n_lim = n_lim + 1 if weak else 0
            if n_lim >= 3:
                frozen, lim_joint = th, (ctl.limited[0] if ctl.limited else None)
                t_end = min(t_end, t + prm["t_hold"])
        th_max = max(th_max, th)
        plant.set_targets(q_cmd)
        t += prm["dt_c"]
        plant.advance(t)
        if frame_cb is not None:
            frame_cb(plant, t, {"phase": "hold" if frozen is not None or th_ref >= goal else
                                ("squeeze" if t < t_sq_end else "turn"), "th_deg": math.degrees(th)})
        if not trace or t - trace[-1]["t"] >= trace_every - 1e-9:
            trace.append({"t": round(t, 3), "th_deg": round(math.degrees(th), 2),
                          "th_ref_deg": round(math.degrees(th_ref), 2), "z_mm": round(1e3 * float(s["p"][2]), 2),
                          "F": {f: round(s["F"][f], 3) for f in FINGERS}, "n": dict(s["n"]),
                          "dist_mm": {f: round(info[f]["dist_mm"], 2) for f in FINGERS},
                          "q_err_deg": [round(float(x), 2) for x in np.degrees(q_cmd - s["q"])]})
        if s["p"][2] < z0 - 0.03 or not np.all(np.isfinite(s["p"])):
            break
    wall = time.perf_counter() - w0
    se = plant.state()
    th_end = ctl.theta(se["R"])
    fingers = sum(1 for f in FINGERS if se["n"][f] > 0)
    dropped = bool(se["p"][2] < z0 - 0.020 or fingers < 2)
    tilt_end = math.asin(max(-1.0, min(1.0, float(se["R"][2, 2]))))
    return {"turn_held3_max_deg": round(math.degrees(th3_max), 2), "turn_held2_max_deg": round(math.degrees(th2_max), 2),
            "first_loss_t_deg": first_loss, "frozen_deg": None if frozen is None else round(math.degrees(frozen), 2),
            "limit_joint": lim_joint,
            "tilt0_deg": round(math.degrees(ctl.tilt0), 2), "tilt_end_deg": round(math.degrees(tilt_end), 2),
            "turn_end_deg": round(math.degrees(th_end), 2), "turn_max_deg": round(math.degrees(th_max), 2),
            "cos_vertical_end": round(float(se["R"][2, 2]), 4), "dropped": dropped, "fingers_end": fingers,
            "held": (not dropped) and fingers >= 2, "F_end": {f: round(se["F"][f], 3) for f in FINGERS},
            "z_end_mm": round(1e3 * float(se["p"][2]), 2), "z_grip_mm": round(1e3 * float(z0), 2),
            "sat_frac": [round(float(x) / max(1, ctl.ticks), 3) for x in ctl.sat],
            "wall_s": round(wall, 2), "t_sim_s": round(t, 3)}, trace


# ---------------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--hands", nargs="+", default=list(RB.HANDS))
    r.add_argument("--sims", nargs="+", default=["mujoco"])
    r.add_argument("--seeds", nargs="+", type=int, default=[0])
    r.add_argument("--plant", default=PLANT)
    r.add_argument("--mu", default="1.0")
    r.add_argument("--set", nargs="*", default=[], help="controller parameters k=v (see DEFAULTS)")
    r.add_argument("--out", type=Path, default=OUT)
    r.add_argument("--traces", action="store_true")
    a = ap.parse_args()
    prm = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=")
        prm[k] = float(v)
    mu = a.mu if a.mu == "scene" else float(a.mu)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for hand in a.hands:
        p_pads, p_mesh, meta, meta_mesh = scenes(hand, a.plant, mu)
        mirror = Mirror(p_mesh)
        for sim in a.sims:
            # Drake takes the tip from the mesh scene (compliant convex); MuJoCo runs the 1 mm pads
            plant = MjPlant(p_pads, meta) if sim == "mujoco" else DrakePlant(p_mesh, meta_mesh)
            for seed in a.seeds:
                row = {"hand": hand, "sim": sim, "seed": seed, "plant": a.plant, "mu": a.mu, "tip": "tpu2.7",
                       "controller": "hom_turn3", **{f"prm_{k}": v for k, v in prm.items()},
                       "when": time.strftime("%Y-%m-%d %H:%M")}
                try:
                    res, trace = rollout(plant, mirror, meta, seed, prm)
                    row.update(res, status="ok")
                    if a.traces:
                        row["trace"] = trace
                except Exception as e:
                    row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
                with open(a.out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                print(f"{hand} {sim:<6} seed {seed}: " + (
                    f"turn {row['turn_end_deg']:+.1f} deg, held3 max {row['turn_held3_max_deg']:+.1f}, held2 max "
                    f"{row['turn_held2_max_deg']:+.1f}, frozen {row['frozen_deg']} at {row['limit_joint']}, held "
                    f"{row['held']}, fingers {row['fingers_end']}, F {row['F_end']}, {row['wall_s']} s wall"
                    if row["status"] == "ok" else row["error"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
