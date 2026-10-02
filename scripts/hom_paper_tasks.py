#!/usr/bin/env python3
"""The simulated tasks of arXiv 2609.25619 (Wang, Oh, Pollard) beyond Exp 1, on the real_v1 (SR2) hand D8 with
sphere-packed MuJoCo fingertips, under the same controller as the screwdriver chain (scripts/hom_chain.py):
generalized contact frames, the relative contact twist and the bounded least squares of the paper's Eq. 2,
pad force through the servos as q_touch + J^T(-F n)/kp + tau_g/kp, 100 Hz control, 1 ms physics.

  exp2   Exp 2, object body twist. Pick off the posts and lift as in the chain; then, palm still, each
         component of the tool's reference twist (along and about the pinch axis, the vertical and the tool
         axis; rotations about the pinch midpoint) is stepped + then - at the paper's 20 mm/s and 40 deg/s.
         Thumb and index track the reference with sticking contacts: in Eq. 2 the tool's reference twist
         replaces its measured one and the separation and slide rows are held at zero.
  exp3   Exp 3, the pusher. Pick with the centre of mass 10 mm from the pinch; the middle closes on the top
         of the far end, 35 mm out; the pinch relaxes to a friction hinge (0.8 N per pad, below the 1.2 N at
         which its friction alone would hold the tool against gravity); the middle tracks the pinch velocity
         s (the paper's object twist about the pinch axis) in +/-10 deg cycles, pushing the end down, while
         gravity brings it back. Then the tripod: pinch 3 N with the middle in place, palm lifts 30 mm.
  wield  The wield. The tool stands in the peg hole of the chain; thumb and index close, roll it about its
         own axis with an object-twist reference (40 deg/s for 0.5 s), release, open 6 mm, return to the
         open posture and close again. Cycles are counted in tool rotation.

    PY=logs/20261001-hom_contact/venv/bin/python
    MUJOCO_GL=egl OMP_NUM_THREADS=1 $PY scripts/hom_paper_tasks.py wield --spec mj:spheres:s1:rs0.75:tr0.02 --film /tmp/w.mp4
    MUJOCO_GL=egl OMP_NUM_THREADS=1 $PY scripts/hom_paper_tasks.py all            # every task x model, films, rows
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_chain as K  # noqa: E402
import hom_contact_rig as H  # noqa: E402
import hom_control as C  # noqa: E402
import hom_hand_brake as B  # noqa: E402

OUT = ROOT / "docs/experiments/20261002-hom_chain"
MEDIA = OUT / "media"
SPEC = "mj:spheres:s1:rs0.75:tr0.02"
FI = {f: i for i, f in enumerate(B.FINGERS)}
STICK = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])     # Eq. 2 weights: separation and slide held, roll and spin free
RATE_LIN, RATE_ANG = 0.020, math.radians(40.0)       # the paper's step magnitudes


def unit(v):
    return v / np.linalg.norm(v)


class Rollout:
    """Plant, mirror and the controller's joint references, advanced one 100 Hz tick at a time."""

    def __init__(self, spec, trial, q_init, palm_init, geom_dirs=None, tool_pose=None, film=None, title="",
                 focus="thumb", wide=None):
        self.spec, self.trial = spec, trial
        self.plant = K.make_plant(spec, trial, q_init, palm_init, geom_dirs)
        if tool_pose is not None:
            self.set_tool(*tool_pose)
        sp = B.parse_spec(spec)
        xml = self.plant.xml if self.plant.sim == "mujoco" else self.plant.xml_mirror
        if self.plant.sim == "mujoco" and sp["model"] == "spheres":
            xml = K.chain_scene("mj:point3", trial)[0]
        self.mir = C.Mirror(xml)
        self.lay = K.world_layout(trial)
        self.kp = B.PLANT["kp"]
        self.dt = 1.0 / K.RATE
        self.q_ref = np.array(q_init, float).copy()
        self.q_home = self.q_ref.copy()
        self.palm = np.array(palm_init, float).copy()
        self.F = {f: 0.0 for f in B.FINGERS}
        self.mode = {f: "free" for f in B.FINGERS}
        self.k = 0
        self.rows, self.frames = [], []
        self.title, self.focus = title, focus
        self.renderer = K.ChainRenderer(self.plant, self.lay, spec, focus=focus, wide=wide) if film else None
        self.perf = {"phys": 0.0, "ctrl": 0.0, "sim": 0.0}
        self.t_ctrl0 = time.perf_counter()
        self.read()

    @property
    def t(self):
        return self.plant.t

    def set_tool(self, pos, R):
        pl = self.plant
        if pl.sim == "mujoco":
            pl.d.qpos[pl.tqa:pl.tqa + 3] = pos
            pl.d.qpos[pl.tqa + 3:pl.tqa + 7] = K.quat_from_R(R)
            pl.d.qvel[pl.tva:pl.tva + 6] = 0.0
            pl.mj.mj_forward(pl.m, pl.d)
        else:
            from pydrake.all import RigidTransform, RotationMatrix
            pl.plant.SetFreeBodyPose(pl.pc, pl.toolb, RigidTransform(RotationMatrix(np.asarray(R, float)),
                                                                     np.asarray(pos, float)))

    def read(self):
        st = self.plant.state()
        self.mir.set_state(st["q"], st["v"], st["tool_pos"], st["tool_quat"], st["tool_v"], st["tool_w"],
                           st["palm_q"], st["palm_v"])
        self.st = st
        self.fr = {f: self.mir.frame(f) for f in B.FINGERS}
        return st

    def pinch_mid(self):
        return 0.5 * (self.fr["thumb"]["p_fing"] + self.fr["index"]["p_fing"])

    def targets(self):
        q = self.q_ref.copy()
        for f in B.FINGERS:
            if self.mode[f] == "force" and self.F[f] != 0.0:
                i, fr = FI[f], self.fr[f]
                self.mir.mj.mj_jac(self.mir.m, self.mir.d, self.mir.jacp, None, fr["p_fing"], self.mir.tip[f])
                J = self.mir.jacp[:, self.mir.fv[f]]
                q[3 * i:3 * i + 3] += J.T @ (-self.F[f] * fr["n"]) / self.kp
        return q + self.mir.gravity() / self.kp

    def tick(self, phase, line=None, log=None):
        self.plant.set_targets(self.targets(), self.palm)
        t1 = time.perf_counter()
        self.perf["ctrl"] += t1 - self.t_ctrl0
        if self.k % 2 == 0:
            cc = self.plant.contacts()
            row = {"t": round(self.t, 3), "phase": phase}
            for f in B.FINGERS:
                row[f"N_{f}"] = round(cc[f]["N"], 4)
                row[f"n_{f}"] = cc[f]["n"]
                row[f"F_{f}"] = round(self.F[f], 3)
            row["N_support"] = round(cc["support"]["N"], 4)
            if log:
                row.update(log)
            self.rows.append(row)
        if self.renderer is not None and self.k % 4 == 0:
            self.frames.append(self.renderer.frame(self.plant, self.st, self.title, phase, None, self.F["thumb"],
                                                   self.fr[self.focus], line=line))
        t2 = time.perf_counter()
        self.plant.step(self.dt)
        t3 = time.perf_counter()
        self.perf["phys"] += t3 - t2
        self.perf["sim"] += self.dt
        self.k += 1
        self.t_ctrl0 = time.perf_counter()
        self.read()

    def hom(self, f, vref, w, twist=None, q_h=None):
        """One Eq. 2 step for finger f. twist = (v, w, c): the tool's reference twist (world frame, v the velocity
        of the point c) stands in for its measured motion; None keeps the measured motion."""
        i = FI[f]
        sl = slice(3 * i, 3 * i + 3)
        A, b, fr = self.mir.twist(f, self.fr[f])
        if twist is not None:
            v, wv, c = twist
            E = fr["E"]
            vo = v + np.cross(wv, fr["p_obj"] - c)
            b = np.concatenate([E.T @ (-wv), E.T @ (-vo) / C.ELL])
        qm = self.st["q"][sl]
        u = C.solve_u(A, b, np.asarray(vref, float), w, qm, self.mir.lo[f], self.mir.hi[f], self.dt,
                      self.q_home[sl] if q_h is None else q_h)
        self.q_ref[sl] = np.clip(self.q_ref[sl] + u * self.dt, qm - C.LEAD, qm + C.LEAD)
        return u

    def close(self, fingers, F_on, v=0.015, ramp=0.25, settle=0.5, timeout=2.5, phase="close", line=None):
        """Fingers approach along their contact normals (v_sep = -v, slide held) and switch to force at 0.3 mm,
        their touching posture corrected by the gap still open; F ramps to F_on. True when all touch."""
        t0 = self.t
        t_c = {}
        for f in fingers:
            self.mode[f] = "approach"
        while True:
            for f in fingers:
                i = FI[f]
                sl = slice(3 * i, 3 * i + 3)
                if self.mode[f] == "approach":
                    vref = np.zeros(6)
                    vref[3] = -v / C.ELL
                    self.hom(f, vref, STICK)
                    fr = self.fr[f]
                    if fr["dist"] <= 0.0003:
                        self.mode[f] = "force"
                        self.mir.mj.mj_jac(self.mir.m, self.mir.d, self.mir.jacp, None, fr["p_fing"], self.mir.tip[f])
                        Jf = self.mir.jacp[:, self.mir.fv[f]]
                        self.q_ref[sl] = self.st["q"][sl] + np.linalg.pinv(Jf) @ (-fr["dist"] * fr["n"])
                        t_c[f] = self.t
                if self.mode[f] == "force":
                    self.F[f] = F_on * min(1.0, (self.t - t_c[f]) / ramp)
            self.tick(phase, line)
            if len(t_c) == len(fingers) and self.t - max(t_c.values()) > settle:
                return True
            if self.t - t0 > timeout:
                return False

    def ramp_force(self, fingers, F_to, T, phase, geometric=True, line=None, log=None):
        F_from = {f: self.F[f] for f in fingers}
        t0 = self.t
        while self.t - t0 < T:
            s = min(1.0, (self.t - t0) / T)
            for f in fingers:
                a, b = F_from[f], F_to
                self.F[f] = a * (b / a) ** s if geometric and a > 0 and b > 0 else a + s * (b - a)
            self.tick(phase, line, log() if log else None)

    def move_joints(self, f, q_to, T, phase, line=None):
        i = FI[f]
        sl = slice(3 * i, 3 * i + 3)
        q_from = self.q_ref[sl].copy()
        t0 = self.t
        while self.t - t0 < T + 0.05:
            s, _, _ = K.minjerk(self.t - t0, T)
            self.q_ref[sl] = q_from + s * (np.asarray(q_to) - q_from)
            self.tick(phase, line)

    def film_out(self, path):
        if self.renderer is not None and path:
            H.write_mp4(self.frames, path, fps=25)


# ------------------------------------------------------------------------------------------- pick

def pick_and_lift(spec, trial, film=None, title="", focus="thumb", wide=None, geom_dirs=None):
    """The chain's pick (approach, close at F0, lift 90 mm), returning the rollout and whether both pads hold."""
    lay = K.world_layout(trial)
    P, a = lay["P"], lay["a"]
    path, _ = K.postures([K.GAP_OPEN] + [float(g) for g in np.linspace(K.GAP_OPEN - 0.001, -K.SQUEEZE_MAX, 12)])
    q_open = path[K.GAP_OPEN]
    perc, d_cg = trial["perc"], trial["d_cg"]
    ax_hat = K.rotz(perc[2]) @ lay["axis"]
    c_hat = lay["c"] + np.array([perc[0], perc[1], 0.0])
    psi = math.atan2(ax_hat[1], ax_hat[0]) - math.atan2(a[1], a[0])
    psi = (psi + math.pi) % (2 * math.pi) - math.pi
    s_hat = c_hat - d_cg * ax_hat
    pick = np.r_[s_hat - K.rotz(psi) @ P, psi]
    start = pick.copy()
    start[2] += 0.060
    if geom_dirs is None and B.parse_spec(spec)["model"] == "spheres":
        geom_dirs = K.contact_dirs(q_open)
    ro = Rollout(spec, trial, q_open, start, geom_dirs=geom_dirs, film=film, title=title, focus=focus, wide=wide)
    t0 = ro.t
    while ro.t - t0 < 1.1:
        s, _, _ = K.minjerk(ro.t - t0, 1.0)
        ro.palm = start + s * (pick - start)
        ro.tick("approach")
    ok = ro.close(("thumb", "index"), K.F0)
    lift_from = ro.palm.copy()
    t0 = ro.t
    while ro.t - t0 < 1.5:
        s, _, _ = K.minjerk(ro.t - t0, 1.2)
        ro.palm = lift_from + s * np.array([0, 0, K.LIFT, 0])
        ro.tick("lift")
    cc = ro.plant.contacts()
    held = bool(cc["thumb"]["n"] > 0 and cc["index"]["n"] > 0 and cc["support"]["N"] < 0.05)
    return ro, bool(ok and held)


def wide_in_hand(trial, el=-3.0, dist=0.24):
    lay = K.world_layout(trial)
    s = lay["s_nom"]
    u = lay["u"]
    return {"lookat": [s[0], s[1], s[2] + K.LIFT - 0.005], "distance": dist,
            "azimuth": math.degrees(math.atan2(u[1], u[0])), "elevation": el}


# -------------------------------------------------------------------------------------------- Exp 2

EXP2_SEQ = ("v_pinch", "v_up", "v_tool", "w_pinch", "w_up", "w_tool")


def exp2(spec=SPEC, film=None, T_step=0.4, T_rest=0.3, w_spin=0.0):
    """w_spin > 0 also holds each pad's spin about its normal at the tool's (Eq. 2 spin row)."""
    t0w = time.time()
    trial = K.make_trial(0)
    title = f"Exp 2, tool twist in the pinch: {K.label(spec)}"
    ro, ok = pick_and_lift(spec, trial, film, title, wide=wide_in_hand(trial))
    res = {"task": "exp2", "spec": spec, "pick_ok": ok, "w_spin": w_spin}
    # task frame at the start: e1 pinch axis (thumb -> index), e3 tool axis, e2 = e3 x e1 turned upward
    e1 = unit(ro.fr["index"]["p_fing"] - ro.fr["thumb"]["p_fing"])
    e3 = ro.st["tool_R"][:, 2]
    e3 = unit(e3 - (e3 @ e1) * e1)
    e2 = np.cross(e3, e1)
    if e2[2] < 0:
        e3, e2 = -e3, -e2
    Ef = np.column_stack([e1, e2, e3])
    sched = []
    for comp in EXP2_SEQ:
        mag = RATE_LIN if comp[0] == "v" else RATE_ANG
        sched += [(comp, +mag, T_step), (comp, -mag, T_step), (comp, 0.0, T_rest)]
    ro.ramp_force(("thumb", "index"), K.F0, 0.3, "settle")
    trace = []
    for comp, val, T in sched:
        k = EXP2_SEQ.index(comp)
        ref = np.zeros(6)
        ref[k] = val
        t0 = ro.t
        while ro.t - t0 < T:
            c = ro.pinch_mid()
            v_ref = Ef @ ref[:3]
            w_ref = Ef @ ref[3:]
            for f in ("thumb", "index"):
                ro.hom(f, np.zeros(6), STICK + np.array([w_spin, 0, 0, 0, 0, 0]), twist=(v_ref, w_ref, c))
            st = ro.st
            v_c = st["tool_v"] + np.cross(st["tool_w"], c - st["tool_pos"])
            ach = np.r_[Ef.T @ v_c, Ef.T @ st["tool_w"]]
            sl = {f: ro.mir.achieved(f, ro.fr[f])[0] for f in ("thumb", "index")}
            slide = max(float(np.linalg.norm(sl[f][4:6])) * C.ELL * 1e3 for f in sl)
            trace.append([round(ro.t, 3), comp, *ref.tolist(), *ach.tolist(), slide])
            lab = f"{comp} {'+' if val > 0 else '-' if val < 0 else ' '}"
            unit_s = "mm/s" if comp[0] == "v" else "deg/s"
            sc = 1e3 if comp[0] == "v" else 180 / math.pi
            ro.tick(f"step {lab}", f"step {lab:9s} ref {val * sc:+5.0f} got {ach[k] * sc:+6.1f} {unit_s}  t {ro.t:5.2f} s",
                    {"comp": comp, "ref": round(val * sc, 2), "ach": round(float(ach[k] * sc), 3)})
    tr = np.array([row[2:15] for row in trace], float)
    ref, ach = tr[:, :6], tr[:, 6:12]
    scale = np.array([1e3] * 3 + [180 / math.pi] * 3)
    err = (ach - ref) * scale
    res["per_comp"] = {}
    for k, comp in enumerate(EXP2_SEQ):
        on = np.abs(ref[:, k]) > 0
        e = err[on]
        lin = np.delete(e[:, :3], k, 1) if k < 3 else e[:, :3]          # the components held at zero
        ang = np.delete(e[:, 3:], k - 3, 1) if k >= 3 else e[:, 3:]
        res["per_comp"][comp] = {"rmse_cmd": float(np.sqrt((e[:, k] ** 2).mean())),
                                 "gain": float(np.median(ach[on, k] / ref[on, k])),
                                 "cross_lin_mmps": float(np.sqrt((lin ** 2).sum(1).mean())),
                                 "cross_ang_dps": float(np.sqrt((ang ** 2).sum(1).mean())),
                                 "slide_rms_mmps": float(np.sqrt((tr[on, 12] ** 2).mean()))}
    ang = np.degrees(np.cumsum(ach[:, 3]) * ro.dt)                  # tool angle about the pinch axis, integrated
    res["pinch_angle_end_deg"] = float(ang[-1])
    res["pinch_angle_max_deg"] = float(ang[np.argmax(np.abs(ang))])
    res["rmse_lin_mmps"] = float(np.sqrt((err[:, :3] ** 2).sum(1).mean()))
    res["rmse_ang_dps"] = float(np.sqrt((err[:, 3:] ** 2).sum(1).mean()))
    cc = ro.plant.contacts()
    res["held_end"] = bool(cc["thumb"]["n"] > 0 and cc["index"]["n"] > 0)
    res.update(_wrap(ro, t0w))
    res["trace_cols"] = ["t", "comp"] + [f"ref_{c}" for c in EXP2_SEQ] + [f"ach_{c}" for c in EXP2_SEQ] + ["slide_mmps"]
    res["trace"] = trace[::2]
    ro.film_out(film)
    return res


# -------------------------------------------------------------------------------------------- Exp 3

def middle_touch_dir(trial, ell):
    """Middle posture touching the top of the tool `ell` beyond the pinch on the side away from the centre of mass
    (palm frame, tool horizontal) and the tip-frame direction of that contact, for the sphere cap."""
    import mujoco
    m, d = K.ik_model()
    P, u, a = K.pinch_frame()
    tgt = P - ell * a + (H.R_TOOL + B.R_TIP) * np.array([0, 0, 1.0])
    best = None
    for seed in ([0.0, 0.6, 0.6], [0.5, 1.0, 0.3], [-0.5, 1.0, 0.3], [0.0, 1.4, 0.2]):
        q, e = B.solve_tip(m, d, "middle", tgt, np.array(seed), iters=300)
        if best is None or e < best[1]:
            best = (q, e)
    q = best[0]
    for n, v in zip(B.JOINTS["middle"], q):
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
    mujoco.mj_kinematics(m, d)
    R = d.xmat[m.body("middle_tip").id].reshape(3, 3)
    return q, R.T @ np.array([0, 0, -1.0]), best[1]


def exp3(spec=SPEC, film=None, d_cg=0.010, ell=0.035, N_hinge=0.8, F_push=0.1, cycles=2, rate_dps=20.0,
         amp_deg=10.0, gap=0.006):
    import mujoco  # noqa: F401
    t0w = time.time()
    trial = K.make_trial(0, d_cg=d_cg)
    q_touch_palm, dir_mid, ik_e = middle_touch_dir(trial, ell)
    dirs = None
    if B.parse_spec(spec)["model"] == "spheres":
        path, _ = K.postures([K.GAP_OPEN])
        dirs = dict(K.contact_dirs(path[K.GAP_OPEN]))
        dirs["middle"] = dir_mid
    title = f"Exp 3, pusher and tripod: {K.label(spec)}"
    ro, ok = pick_and_lift(spec, trial, film, title, focus="middle", wide=wide_in_hand(trial), geom_dirs=dirs)
    res = {"task": "exp3", "spec": spec, "pick_ok": ok, "d_cg_mm": d_cg * 1e3, "ell_mm": ell * 1e3,
           "N_hinge": N_hinge, "F_push": F_push, "ik_err_mm": ik_e * 1e3}
    # pre-contact posture from the measured tool pose
    st = ro.st
    pm = ro.pinch_mid()
    ax = st["tool_R"][:, 2]                                    # toward the centre-of-mass side
    up = np.array([0, 0, 1.0])
    n_top = unit(up - (up @ ax) * ax)
    p_c = pm - ell * ax + H.R_TOOL * n_top
    q_pre, e_pre = B.solve_tip(ro.mir.m, ro.mir.d, "middle", p_c + (B.R_TIP + gap) * n_top,
                               np.array(q_touch_palm), iters=300)
    res["pre_ik_err_mm"] = e_pre * 1e3
    ro.read()
    ro.move_joints("middle", q_pre, 1.0, "pusher in")
    ro.q_home[6:9] = q_pre
    res["pusher_closed"] = ro.close(("middle",), F_push, v=0.010, ramp=0.2, settle=0.3, phase="pusher close")
    # the pinch becomes a hinge
    ro.ramp_force(("thumb", "index"), N_hinge, 0.6, "hinge")
    t0 = ro.t
    while ro.t - t0 < 0.4:
        ro.tick("hinge")
    # pinch axis and the sign that turns the far end away from the pusher
    pm = ro.pinch_mid()
    u_p = unit(ro.fr["index"]["p_fing"] - ro.fr["thumb"]["p_fing"])
    if np.cross(u_p, ro.fr["middle"]["p_obj"] - pm) @ ro.fr["middle"]["n"] > 0:
        u_p = -u_p
    a0 = ro.st["tool_R"][:, 2].copy()
    w_dir = np.cross(u_p, a0)

    def angle():
        a = ro.st["tool_R"][:, 2]
        return math.degrees(math.atan2(a @ w_dir, a @ a0))

    T_half = amp_deg / rate_dps
    sched = []
    for _ in range(cycles):
        sched += [(+1, T_half), (0, 0.3), (-1, 2 * T_half), (0, 0.3), (+1, T_half), (0, 0.3)]
    trace = []
    phi_ref = 0.0
    for sgn, T in sched:
        t0 = ro.t
        while ro.t - t0 < T:
            s_ref = sgn * math.radians(rate_dps)
            pm = ro.pinch_mid()
            ro.hom("middle", np.zeros(6), STICK, twist=(np.zeros(3), s_ref * u_p, pm))
            s_meas = float(ro.st["tool_w"] @ u_p)
            ach, _ = ro.mir.achieved("middle", ro.fr["middle"])
            slide = float(np.linalg.norm(ach[4:6])) * C.ELL * 1e3
            roll = float(np.linalg.norm(ach[1:3]))
            phi = angle()
            trace.append([round(ro.t, 3), s_ref, s_meas, phi, phi_ref, slide, roll, ro.fr["middle"]["dist"]])
            ro.tick("rotate", f"rotate  s ref {math.degrees(s_ref):+4.0f} got {math.degrees(s_meas):+6.1f} deg/s  "
                              f"angle {phi:+5.1f}  t {ro.t:5.2f} s",
                    {"s_ref": round(math.degrees(s_ref), 2), "s": round(math.degrees(s_meas), 3), "phi": round(phi, 3)})
            phi_ref += math.degrees(s_ref) * ro.dt
    tr = np.array(trace)
    on = tr[:, 1] != 0
    err = np.degrees(tr[:, 2] - tr[:, 1])
    res["rmse_s_dps"] = float(np.sqrt((err ** 2).mean()))
    res["rmse_s_on_dps"] = float(np.sqrt((err[on] ** 2).mean()))
    res["gain_s"] = float(np.median(tr[on, 2] / tr[on, 1]))
    res["phi_range_deg"] = [float(tr[:, 3].min()), float(tr[:, 3].max())]
    res["phi_track_rmse_deg"] = float(np.sqrt(((tr[:, 3] - tr[:, 4]) ** 2).mean()))
    res["pusher_slide_rms_mmps"] = float(np.sqrt((tr[on, 5] ** 2).mean()))
    res["pusher_roll_rms_dps"] = float(np.degrees(np.sqrt((tr[on, 6] ** 2).mean())))
    res["pusher_gap_max_mm"] = float(tr[:, 7].max() * 1e3)
    # tripod: pinch firm with the middle in place, then the palm lifts 30 mm
    phi_t0 = angle()
    ro.ramp_force(("thumb", "index"), 3.0, 0.4, "tripod")
    t0 = ro.t
    while ro.t - t0 < 0.3:
        ro.tick("tripod")
    lift_from = ro.palm.copy()
    t0 = ro.t
    while ro.t - t0 < 1.3:
        s, _, _ = K.minjerk(ro.t - t0, 1.0)
        ro.palm = lift_from + s * np.array([0, 0, 0.030, 0])
        ro.tick("tripod lift", f"tripod lift  angle {angle():+5.1f} deg  t {ro.t:5.2f} s")
    cc = ro.plant.contacts()
    res["tripod_held"] = bool(all(cc[f]["n"] > 0 for f in B.FINGERS))
    res["tripod_phi_drift_deg"] = float(angle() - phi_t0)
    res["tripod_N"] = {f: cc[f]["N"] for f in B.FINGERS}
    res.update(_wrap(ro, t0w))
    res["trace_cols"] = ["t", "s_ref", "s", "phi", "phi_ref", "slide_mmps", "roll_rps", "gap_m"]
    res["trace"] = tr[::2].round(6).tolist()
    ro.film_out(film)
    return res


# -------------------------------------------------------------------------------------------- wield

def wield(spec=SPEC, film=None, cycles=6, F_w=2.0, rate_dps=40.0, T_twist=0.5, open_mm=6.0):
    t0w = time.time()
    trial = K.make_trial(0)
    lay = K.world_layout(trial)
    P, u = lay["P"], lay["u"]
    hole = lay["hole"]
    path, _ = K.postures([K.GAP_OPEN])
    q_open = path[K.GAP_OPEN]
    tool_pos = np.array([hole[0], hole[1], H.HL_TOOL + 0.0005])
    R_tool = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])       # body z down: the marked end in the hole
    grip = tool_pos + np.array([0, 0, trial["d_cg"]])
    palm0 = np.r_[grip - P, 0.0]
    wide = {"lookat": [hole[0], hole[1], 0.055], "distance": 0.20, "elevation": -32.0,
            "azimuth": math.degrees(math.atan2(u[1], u[0])) + 35.0}
    title = f"Wield, retract-turn cycles in the peg hole: {K.label(spec)}"
    ro = Rollout(spec, trial, q_open, palm0, tool_pose=(tool_pos, R_tool), film=film, title=title, wide=wide)
    res = {"task": "wield", "spec": spec, "cycles": cycles, "F_w": F_w, "rate_dps": rate_dps, "T_twist": T_twist}

    def theta():
        R = ro.st["tool_R"]
        return math.atan2(R[1, 0], R[0, 0])

    th = {"last": theta(), "acc": 0.0}

    def turned():
        t_ = theta()
        dth = (t_ - th["last"] + math.pi) % (2 * math.pi) - math.pi
        th["acc"] += dth
        th["last"] = t_
        return math.degrees(th["acc"])

    def tilt():
        return math.degrees(math.acos(min(1.0, abs(ro.st["tool_R"][2, 2]))))

    def line(ph, k):
        return f"cycle {k}  {ph:7s} turned {turned():+6.1f} deg  tilt {tilt():3.1f}  t {ro.t:5.2f} s"

    t0 = ro.t
    while ro.t - t0 < 0.3:
        ro.tick("settle", line("settle", 0))
    ok0 = ro.close(("thumb", "index"), F_w, line=line("close", 0))
    res["first_close_ok"] = ok0
    per_cycle, trace = [], []
    for k in range(1, cycles + 1):
        a0 = turned()
        t0 = ro.t
        while ro.t - t0 < T_twist:
            ax_up = -ro.st["tool_R"][:, 2]
            w_ref = math.radians(rate_dps) * ax_up
            for f in ("thumb", "index"):
                ro.hom(f, np.zeros(6), STICK, twist=(np.zeros(3), w_ref, ro.st["tool_pos"]))
            w_meas = float(ro.st["tool_w"] @ ax_up)
            sl = max(float(np.linalg.norm(ro.mir.achieved(f, ro.fr[f])[0][4:6])) * C.ELL * 1e3 for f in ("thumb", "index"))
            trace.append([round(ro.t, 3), k, rate_dps, math.degrees(w_meas), turned(), tilt(), sl])
            ro.tick("twist", line("twist", k), {"cycle": k, "w": round(math.degrees(w_meas), 2), "turned": round(turned(), 3)})
        a1 = turned()
        ro.ramp_force(("thumb", "index"), 0.0, 0.15, "release", geometric=False, line=line("release", k))
        for f in ("thumb", "index"):
            ro.mode[f] = "free"
        t0 = ro.t
        while ro.t - t0 < open_mm * 1e-3 / 0.015:
            vref = np.zeros(6)
            vref[3] = 0.015 / C.ELL
            for f in ("thumb", "index"):
                ro.hom(f, vref, STICK)
            ro.tick("open", line("open", k))
        a2 = turned()
        q_back = q_open.copy()
        q_from = ro.q_ref.copy()
        t0 = ro.t
        while ro.t - t0 < 0.65:
            s, _, _ = K.minjerk(ro.t - t0, 0.6)
            ro.q_ref[:6] = q_from[:6] + s * (q_back[:6] - q_from[:6])
            ro.tick("return", line("return", k))
        closed = ro.close(("thumb", "index"), F_w, line=line("close", k))
        a3 = turned()
        per_cycle.append({"cycle": k, "twist_deg": a1 - a0, "release_open_deg": a2 - a1, "return_close_deg": a3 - a2,
                          "net_deg": a3 - a0, "closed": closed, "tilt_deg": tilt()})
    tr = np.array(trace, float)
    err = tr[:, 3] - tr[:, 2]
    res["per_cycle"] = per_cycle
    res["turned_deg"] = turned()
    res["twist_cmd_deg"] = rate_dps * T_twist
    res["rmse_w_dps"] = float(np.sqrt((err ** 2).mean()))
    res["gain_w"] = float(np.median(tr[:, 3] / tr[:, 2]))
    res["slide_rms_mmps"] = float(np.sqrt((tr[:, 6] ** 2).mean()))
    res["tilt_max_deg"] = float(tr[:, 5].max())
    cc = ro.plant.contacts()
    res["held_end"] = bool(cc["thumb"]["n"] > 0 and cc["index"]["n"] > 0)
    res.update(_wrap(ro, t0w))
    res["trace_cols"] = ["t", "cycle", "w_ref", "w", "turned", "tilt", "slide_mmps"]
    res["trace"] = tr[::2].round(5).tolist()
    ro.film_out(film)
    return res


def _wrap(ro, t0w):
    return {"wall_s": time.time() - t0w, "sim_s": ro.t, "perf": ro.perf, "rows": ro.rows[::2],
            "info": getattr(ro.plant, "info", {}), "label": K.label(ro.spec)}


# ------------------------------------------------------------------------------------------- driver

TASKS = {"exp2": exp2, "exp3": exp3, "wield": wield}


def tiles(keys=("s1", "p4s", "mp3", "dhy")):
    """Per task, the wide halves of the four contact models' films in a 2 x 2 grid, and a poster frame."""
    import imageio.v2 as imageio
    from PIL import Image
    for task in TASKS:
        fr = {k: [f[:, :480] for f in imageio.get_reader(str(MEDIA / f"20261002-paper_{task}_{k}.mp4"))] for k in keys}
        n = max(len(v) for v in fr.values())
        get = lambda k, i: fr[k][min(i, len(fr[k]) - 1)]  # noqa: E731
        grid = [np.concatenate([np.concatenate([get(k, i) for k in keys[:2]], 1),
                                np.concatenate([get(k, i) for k in keys[2:]], 1)], 0) for i in range(n)]
        out = MEDIA / f"20261002-paper_{task}_models.mp4"
        imageio.mimsave(str(out), grid, fps=25, macro_block_size=1, codec="libx264", quality=None,
                        ffmpeg_params=["-crf", "27", "-preset", "slow"])
        Image.fromarray(grid[min(n - 1, int(0.6 * n))]).save(MEDIA / f"20261002-paper_{task}_models_poster.jpg", quality=85)
        print("tile", task, n, round(out.stat().st_size / 1e6, 2), "MB", flush=True)
MODELS = {"s1": SPEC, "p4s": "mj:point4s", "mp3": "mj:point3", "dhy": "drake:hydro:rt0.01", "s1spin": SPEC}
VARIANT = {"s1spin": {"exp2": {"w_spin": 1.0}}}            # Eq. 2 also holds each pad's spin at the tool's


def run_all(keys=("s1", "p4s", "mp3"), tasks=("wield", "exp3", "exp2")):
    MEDIA.mkdir(parents=True, exist_ok=True)
    p = OUT / "paper_tasks.jsonl"
    have = set()
    if p.exists():
        have = {(json.loads(l)["task"], json.loads(l)["key"]) for l in open(p) if l.strip()}
    for task in tasks:
        for key in keys:
            spec = MODELS[key]
            kw = VARIANT.get(key, {}).get(task)
            if (task, key) in have or (key in VARIANT and kw is None):
                continue
            film = MEDIA / f"20261002-paper_{task}_{key}.mp4"
            r = TASKS[task](spec, film=str(film), **(kw or {}))
            r["film"], r["key"] = str(film.relative_to(OUT)), key
            H.append_row(p, r)
            print(task, key, {k: v for k, v in r.items() if k not in ("trace", "rows", "per_comp", "per_cycle", "perf",
                                                                     "info", "trace_cols")}, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=sorted(TASKS) + ["all", "tiles"])
    ap.add_argument("--spec", default=SPEC)
    ap.add_argument("--film")
    ap.add_argument("--models", default="s1,p4s,mp3")
    ap.add_argument("--tasks", default="wield,exp3,exp2")
    args = ap.parse_args()
    if args.task == "all":
        run_all(tuple(args.models.split(",")), tuple(args.tasks.split(",")))
        return
    if args.task == "tiles":
        tiles()
        return
    r = TASKS[args.task](args.spec, film=args.film)
    print(json.dumps({k: v for k, v in r.items() if k not in ("trace", "rows")}, default=H._json_default, indent=1))


if __name__ == "__main__":
    main()
