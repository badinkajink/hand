#!/usr/bin/env python3
"""Simulator of the fingertip's calibration session (ur5_tip_calibration.py --sim): the UR5 CB3, the OptoForce HEX-E, the
board and the pad on the printed indenter, behind the same calls as the hardware, on a virtual clock that the session's
sleeps advance.

Robot: the CB3's DH parameters (UR's, as in ur_description's ur5.urdf: d1 89.159, a2 -425, a3 -392.25, d4 109.15, d5
94.65, d6 82.3 mm); the controller reports the flange of the nominal parameters, the tool sits where perturbed ones put it
(Truth.dh_err), with a repeatability offset drawn at every motion. Moves follow trapezoidal speed profiles along their path
(rotations count 100 mm per radian), stopL brakes with its deceleration.

Contact: the pad's linear FE model (scripts/fingertip_wrench_fe.py --surface): the surface compliance between its 812
points (disp_u, the displacement of every point under 1 N at each), scaled by Truth.pad_soft, with the HEX-E's datasheet
compliance (Fxy 1.7 mm, Fz 0.3 mm, Txy 2.5 deg, Tz 5 deg at 200 N, 10 and 6.5 N m) in series as a rigid motion of the
pad; normal contact with the sphere (or the rod) as a linear complementarity problem on the points near it, friction
(Truth.mu) for the motion of the indenter along the pad since the contact began, less its motion along the press.

HEX-E: the contact wrench and the payload's weight about the sensor's origin, through its axes (Truth.hex_turn_deg about
the tool axis, sign), crosstalk, offsets that drift, and its datasheet noise (1 sigma: 0.035, 0.035, 0.15 N, 0.002,
0.002, 0.001 N m) at 500 Hz. Board: the magnet model of the as-found calibration (scripts/fingertip_wrench_model.py) with
the FE's magnet motions for the contact forces, Earth's field turned into the pad's frame, the chips' gain falling with
their warm-up (0.65-0.70 % per C, 1.3 C over the first minutes) and their noise, 140 records/s.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_tip_calibration as C  # noqa: E402

BASIS = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-b_v2_surface_basis.npz"
BOARD_CAL = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-calib_pad_as_found.json"
DH_CB3 = dict(d=[89.159, 0.0, 0.0, 109.15, 94.65, 82.3], a=[0.0, -425.0, -392.25, 0.0, 0.0, 0.0],
              alpha=[math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0])
HEX_COMPLIANCE = dict(f_mm_N=(1.7 / 200, 1.7 / 200, 0.3 / 200), r_rad_Nmm=(math.radians(2.5) / 1e4, math.radians(2.5) / 1e4,
                                                                          math.radians(5.0) / 6.5e3))
HEX_NOISE = (0.035, 0.035, 0.15, 0.002, 0.002, 0.001)


# ------------------------------------------------------------------------------------------- the CB3
def fk(q, dh=DH_CB3, frames=False):
    T = np.eye(4)
    out = [T.copy()]
    for i in range(6):
        th = q[i] + dh.get("off", [0.0] * 6)[i]
        ct, st = math.cos(th), math.sin(th)
        ca, sa = math.cos(dh["alpha"][i]), math.sin(dh["alpha"][i])
        A = np.array([[ct, -st * ca, st * sa, dh["a"][i] * ct], [st, ct * ca, -ct * sa, dh["a"][i] * st],
                      [0.0, sa, ca, dh["d"][i]], [0.0, 0.0, 0.0, 1.0]])
        T = T @ A
        out.append(T.copy())
    return out if frames else T


def jac(q, dh=DH_CB3):
    Fs = fk(q, dh, frames=True)
    p = Fs[-1][:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        z, o = Fs[i][:3, 2], Fs[i][:3, 3]
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J


def ik(T, q0, dh=DH_CB3, iters=60, tol=1e-7):
    """The solution nearest q0, as the controller's get_inverse_kin returns it: damped least squares from q0, and when
    that fails from q0 with the shoulder, elbow and wrist branches flipped; q or None."""
    q = _ik1(T, q0, dh, iters, tol)
    if q is not None:
        return q
    q0 = np.asarray(q0, float)
    best = None
    for d1 in (0.0, math.pi, -math.pi):
        for d3 in (0.0, -2 * q0[2]):
            for d5 in (0.0, -2 * q0[4]):
                for d6 in (0.0, math.pi, -math.pi):
                    s = q0 + np.array([d1, 0.0, d3, 0.0, d5, d6])
                    qq = _ik1(T, s, dh, iters, tol)
                    if qq is not None:
                        dist = float(np.abs(qq - q0).max())
                        if best is None or dist < best[0]:
                            best = (dist, qq)
    return None if best is None else best[1]


def _ik1(T, q0, dh=DH_CB3, iters=60, tol=1e-7):
    q = np.array(q0, float)
    for _ in range(iters):
        Tc = fk(q, dh)
        dp = T[:3, 3] - Tc[:3, 3]
        dw = C.log_so3(T[:3, :3] @ Tc[:3, :3].T)
        e = np.r_[dp, dw * 200.0]
        if np.linalg.norm(dp) < tol and np.linalg.norm(dw) < tol / 100:
            return q
        J = jac(q, dh)
        J[3:] *= 200.0
        lam = 1e-4 if np.linalg.norm(e) < 10 else 1e-2
        q = q + np.linalg.solve(J.T @ J + lam * np.eye(6), J.T @ e)
    Tc = fk(q, dh)
    if np.linalg.norm(T[:3, 3] - Tc[:3, 3]) < 1e-4 and np.linalg.norm(C.log_so3(T[:3, :3] @ Tc[:3, :3].T)) < 1e-6:
        return q
    return None


# ------------------------------------------------------------------------------------------- what the session must find
@dataclass
class Truth:
    em_err: tuple = (0.0012, -0.0009, 0.0157, 0.25, -0.20, 0.45)     # mount on the flange: rad (x, y, z_M), mm
    mp_err: tuple = (0.0050, -0.0035, 0.0070, 0.15, -0.12, 0.20)     # pad on the mount (seat)
    indenter_err: tuple = (0.0, 0.0, 0.0035, 1.5, -2.0, 0.8)         # indenter against the plan's placement (yaw rad, mm)
    level_err: tuple = (0.004, -0.003)                               # plywood tilt (rad about x_B, y_B)
    hex_turn_deg: float = 7.3                                        # the HEX-E's axes about the tool axis
    hex_sign: float = 1.0
    hex_height_err_mm: float = 0.3
    crosstalk: float = 0.012                                         # rms of the off-diagonal terms of its 3 x 3 maps
    hex_offset_N: tuple = (0.8, -1.3, 2.1, 0.03, -0.05, 0.01)
    hex_drift_N_rt_s: float = 0.0015                                 # random walk of each force offset
    payload_kg: float = 0.184                                        # past the HEX-E's sensing element
    payload_com_mm: tuple = (0.2, 2.0, 16.9)                         # from the HEX-E's origin, flange axes
    seat_err_mm: tuple = (0.10, 0.03, 0.04)                          # homing: along the key (its relieved ends), across, depth
    dh_err: float = 0.08                                             # mm and mrad of the DH perturbation
    repeat_mm: float = 0.01
    pad_soft: float = 1.15                                           # the printed pad's compliance over the FE's
    mu: float = 1.0
    earth_uT: tuple = (20.0, 2.0, -46.0)
    warm_C: float = 1.3
    warm_tau_s: float = 200.0
    seat_shift_um: float = 0.0                                       # a rigid shift of the pad after presses over 3 N
    seed: int = 0


class World:
    def __init__(self, plan, truth: Truth, t0=None):
        self.plan, self.tr = plan, truth
        self.rng = np.random.default_rng(truth.seed)
        self.geo = C.Geometry(plan)
        self.t = float(t0 if t0 is not None else time.time())
        self.t_start = self.t
        g = self.geo
        e = truth
        self.T_EM = C.perturb(g.T_EM0, np.r_[e.em_err[:3], e.em_err[3:]])
        self.T_EM[:3, 3] += self.T_EM[:3, :3] @ np.array([0.0, 0.0, e.hex_height_err_mm])
        self.T_MP = C.perturb(g.T_MP0, np.r_[e.mp_err[:3], e.mp_err[3:]])
        self.T_EP = self.T_EM @ self.T_MP
        R_ES = C.rot(C.EZ, e.hex_turn_deg)
        self.T_ES = C.Tm(R_ES, g.T_ES0[:3, 3] + np.array([0.0, 0.0, e.hex_height_err_mm]))
        X = self.rng.normal(0.0, e.crosstalk, (2, 3, 3))
        self.Kf = np.eye(3) + X[0] - np.diag(np.diag(X[0]))
        self.Kt = np.eye(3) + X[1] - np.diag(np.diag(X[1]))
        self.off = np.array(e.hex_offset_N, float)
        Tb = np.array(plan["sim"]["T_BI"], float)
        dI = np.array(e.indenter_err, float)
        Tlev = C.Tm(C.exp_so3([e.level_err[0], e.level_err[1], 0.0]))
        self.T_BI = Tlev @ C.Tm(C.rot(C.EZ, math.degrees(dI[2])), np.r_[dI[3:5], dI[5]]) @ Tb
        self.c_B = C.apply(self.T_BI, g.c_I)
        rd = self.rng.normal(0.0, e.dh_err, (4, 6))
        self.dh_true = dict(d=list(np.array(DH_CB3["d"]) + rd[0] * 1.0), a=list(np.array(DH_CB3["a"]) + rd[1] * 1.0),
                            alpha=list(np.array(DH_CB3["alpha"]) + rd[2] * 1e-3), off=list(rd[3] * 1e-3))
        self.rep = np.zeros(3)
        # pad model
        B = np.load(BASIS)
        self.X, self.N = B["pts"].astype(float), B["normals"].astype(float)
        n = len(self.X)
        du = B["disp_u"].astype(np.float32)                         # [load i, dir a, node j, dir b] um / N
        Cm = np.transpose(du, (2, 3, 0, 1)).reshape(3 * n, 3 * n).astype(float) * 1e-3 * e.pad_soft   # u_j = C F_i, mm/N
        self.Cfe = 0.5 * (Cm + Cm.T)
        self.mot = B["motions"].astype(float) * e.pad_soft          # [i, a, magnet, 6] um, mrad per N
        t1 = np.cross(np.where(np.abs(self.N[:, 2:3]) > 0.9, [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]), self.N)
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        self.Tg = np.stack([t1, np.cross(self.N, t1)], axis=2)      # (n, 3, 2) tangent bases
        self.o_P = C.apply(C.inv(self.T_EP) @ self.T_ES, np.zeros(3))   # the HEX-E's origin in P
        R_PS = (C.inv(self.T_EP) @ self.T_ES)[:3, :3]
        cw = np.r_[HEX_COMPLIANCE["r_rad_Nmm"], HEX_COMPLIANCE["f_mm_N"]]
        Cw = np.zeros((6, 6))
        Cw[:3, :3] = R_PS @ np.diag(cw[:3]) @ R_PS.T
        Cw[3:, 3:] = R_PS @ np.diag(cw[3:]) @ R_PS.T
        self.Cw = Cw
        self.anchor = None
        self.cache = None
        self.t_solve = -1.0
        self.F = np.zeros((0, 3))
        self.Fidx = np.zeros(0, int)
        self.wrench_P = np.zeros(6)
        self.truth_row = np.zeros(12)                                    # f (3, P), centre of pressure (3), normal (3), torsion
        self.shift = np.zeros(3)                                         # (N mm), contact points, kind (1 pad, 2 face)
        # board
        import fingertip_wrench_model as W
        self.Wm = W
        self.cal = W.Calib.load(BOARD_CAL)
        self.model = W.Model(self.cal)
        self.kappa = np.array(W.load_thermal(), float) if Path(W.THERMAL).exists() else np.full(5, -0.0067)
        self.T_amb = 24.0 + self.rng.normal(0.0, 0.2, 5)
        self.peak_N = 0.0
        # robot
        self.q = None
        self.seg = None
        self.teach = False
        self.prot = False
        self.ft_next = self.board_next = self.robot_next = self.t
        self.streams = None

    # ---------------------------------------------------------------- robot motion
    def flange_true(self):
        T = fk(self.q, self.dh_true)
        T = T.copy()
        T[:3, 3] += self.rep
        return T

    def start_segment(self, seg):
        if not np.any(self.wrench_P[:3]):          # a move from out of contact lands repeat_mm (1 sigma) off; in contact
            self.rep = self.rng.normal(0.0, self.tr.repeat_mm, 3)   # the arm keeps its error (it changes smoothly)
        self.seg = seg
        seg.update(s=0.0, v=0.0)

    def seg_pose(self, seg, s):
        if seg["kind"] == "J":
            return seg["q0"] + (seg["q1"] - seg["q0"]) * (s / seg["L"] if seg["L"] > 0 else 1.0)
        cum, Ts = seg["cum"], seg["Ts"]
        k = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(Ts) - 2))
        u = 0.0 if cum[k + 1] == cum[k] else (s - cum[k]) / (cum[k + 1] - cum[k])
        A, B_ = Ts[k], Ts[k + 1]
        w = C.log_so3(A[:3, :3].T @ B_[:3, :3])
        return C.Tm(A[:3, :3] @ C.exp_so3(w * u), A[:3, 3] + (B_[:3, 3] - A[:3, 3]) * u)

    def path_segment(self, Ts, v_mm, a_mm):
        Ts = [self.cmd_pose()] + list(Ts)
        cum = [0.0]
        for A, B_ in zip(Ts[:-1], Ts[1:]):
            cum.append(cum[-1] + max(np.linalg.norm(B_[:3, 3] - A[:3, 3]),
                                     100.0 * np.linalg.norm(C.log_so3(A[:3, :3].T @ B_[:3, :3]))))
        return dict(kind="L", Ts=Ts, cum=np.array(cum), L=cum[-1], vmax=v_mm, amax=a_mm)

    def cmd_pose(self):
        return fk(self.q)

    def step_robot(self, h):
        sg = self.seg
        if sg is None:
            return
        L, s, v = sg["L"], sg["s"], sg["v"]
        brake = v * v / (2 * sg["amax"])
        if L - s <= brake + 1e-12:
            v = max(v - sg["amax"] * h, 0.0)
            if v == 0.0:
                s = L
        else:
            v = min(v + sg["amax"] * h, sg["vmax"])
        s = min(s + v * h, L)
        sg["s"], sg["v"] = s, v
        if sg["kind"] == "J":
            self.q = self.seg_pose(sg, s)
        else:
            q = ik(self.seg_pose(sg, s), self.q)
            if q is None:
                self.prot = True
                self.seg = None
                return
            self.q = q
        if s >= L - 1e-12:
            self.seg = None

    def stop_segment(self, a_mm):
        sg = self.seg
        if sg is None:
            return
        sg["L"] = min(sg["L"], sg["s"] + sg["v"] ** 2 / (2 * a_mm))
        sg["amax"] = a_mm

    # ---------------------------------------------------------------- contact
    def contact(self):
        """Nodal contact forces (pad frame) for the present pose; cached while the pad and the indenter stay put."""
        if self.cache is not None and self.t - self.t_solve < 0.0015:
            return                                                     # the board's sample takes the HEX-E's solve
        T_BP = self.flange_true() @ self.T_EP
        T_PB = C.inv(T_BP)
        key = np.round(T_BP[:3, :], 7).tobytes()
        self.t_solve = self.t
        if self.cache is not None and self.cache[0] == key:
            return
        g = self.geo
        X = self.X + self.shift
        if g.kind == "sphere":
            c = C.apply(T_PB, self.c_B)
            r = X - c
        else:
            a = T_PB[:3, :3] @ (self.T_BI[:3, :3] @ g.rod_axis)
            c = C.apply(T_PB, C.apply(self.T_BI, g.c_I))
            r = X - c
            r = r - (r @ a)[:, None] * a[None]
        dist = np.linalg.norm(r, axis=1)
        gap = dist - g.R
        nh = r / dist[:, None]
        cand = np.nonzero((gap < 0.6) & (np.einsum("ij,ij->i", nh, self.N) < -0.2))[0]
        if g.kind == "rod":
            yl = (X - c) @ a
            cand = cand[np.abs(yl[cand]) < 10.0]
        if len(cand) == 0 or gap[cand].min() >= 0:
            self.F, self.Fidx, self.wrench_P = np.zeros((0, 3)), np.zeros(0, int), np.zeros(6)
            self.truth_row = np.zeros(12)
            self.anchor = None
            self.cache = (key,)
            if g.kind == "sphere":
                self.face_contact(T_PB)
            return
        T_PI = T_PB @ self.T_BI
        if self.anchor is None:
            self.anchor = T_PI.copy()
        K = cand
        idx = (3 * K[:, None] + np.arange(3)[None]).ravel()
        Ck = self.Cfe[np.ix_(idx, idx)]
        G = np.zeros((6, 3 * len(K)))
        for j, i in enumerate(K):
            G[:3, 3 * j:3 * j + 3] = C.skew(X[i] - self.o_P)
            G[3:, 3 * j:3 * j + 3] = np.eye(3)
        Ck = Ck + G.T @ self.Cw @ G
        Nk = np.zeros((3 * len(K), len(K)))
        Tk = np.zeros((3 * len(K), 2 * len(K)))
        for j, i in enumerate(K):
            Nk[3 * j:3 * j + 3, j] = nh[i]
            Tk[3 * j:3 * j + 3, 2 * j:2 * j + 2] = self.Tg[i]
        # the indenter's motion along the pad since contact began, less its motion along the press
        D = T_PI @ C.inv(self.anchor)
        Dinv = C.inv(D)
        x0 = C.apply(Dinv, X[K])
        rel = X[K] - x0
        n_press = -self.N[K[np.argmin(gap[K])]]
        rel = rel - (rel @ n_press)[:, None] * n_press[None]
        s_t = np.einsum("kij,ki->kj", self.Tg[K], rel).ravel()
        q = np.zeros(2 * len(K))
        p = np.zeros(len(K))
        mu = self.tr.mu
        for _ in range(4):
            Mn = Nk.T @ Ck @ Nk
            gk = gap[K] + Nk.T @ Ck @ (Tk @ q)
            Mn = 0.5 * (Mn + Mn.T) + 1e-9 * np.eye(len(K))
            Lc = np.linalg.cholesky(Mn)
            p, _ = nnls(Lc.T, -np.linalg.solve(Lc, gk))
            act = p > 1e-9
            if np.abs(s_t).max() < 1e-7 or not act.any():
                q[:] = 0.0
                break
            stick = np.repeat(act, 2)
            slip = np.zeros(len(K), bool)
            for _ in range(10):
                st = stick & ~np.repeat(slip, 2)
                qs = q.copy()
                qs[~np.repeat(act, 2)] = 0.0
                if st.any():
                    A = Tk[:, st].T @ Ck @ Tk[:, st]
                    rhs = s_t[st] - Tk[:, st].T @ Ck @ (Nk @ p + Tk[:, ~st] @ qs[~st])
                    qs[st] = np.linalg.solve(A + 1e-9 * np.eye(A.shape[0]), rhs)
                qq = qs.reshape(-1, 2)
                mag = np.linalg.norm(qq, axis=1)
                over = act & ~slip & (mag > mu * p + 1e-12)
                q = qs
                if not over.any():
                    break
                slip |= over
                for j in np.nonzero(slip)[0]:
                    m = np.linalg.norm(q[2 * j:2 * j + 2])
                    if m > 0:
                        q[2 * j:2 * j + 2] *= mu * p[j] / m
        F = (Nk @ p + Tk @ q).reshape(-1, 3)
        keep = np.linalg.norm(F, axis=1) > 1e-9
        self.F, self.Fidx = F[keep], K[keep]
        f = F.sum(axis=0)
        tau = np.cross(X[K] - self.o_P, F).sum(axis=0)
        self.wrench_P = np.r_[f, tau]                                  # N, N mm about the HEX-E's origin
        w = p / max(p.sum(), 1e-12)
        cop = (X[K] * w[:, None]).sum(axis=0)
        nc = (self.N[K] * w[:, None]).sum(axis=0)
        nc /= np.linalg.norm(nc)
        tn = float(np.cross(X[K] - cop, F).sum(axis=0) @ nc)
        self.truth_row = np.r_[f, cop, nc, tn, float(keep.sum()), 1.0]
        self.cache = (key,)
        fn = float(np.linalg.norm(f))
        if fn > self.peak_N:
            self.peak_N = fn
        if fn > 60.0:
            self.prot = True

    def face_contact(self, T_PB):
        """The sphere on the mount column's flat palmar face (x_M = 17): a stiff contact (PLA on PLA, 0.004 mm/N) in
        series with the HEX-E's compliance at the contact; the force acts along -x_M on the mount."""
        T_MP = self.T_MP
        c_P = C.apply(T_PB, self.c_B)
        c_M = C.apply(T_MP, c_P)
        gap = c_M[0] - self.geo.face_d - self.geo.R
        if gap >= 0 or not (-22.0 < c_M[1] < 14.0 and 28.0 < c_M[2] < 73.0):
            return
        x_P = C.apply(C.inv(T_MP), np.array([self.geo.face_d, c_M[1], c_M[2]]))
        n_P = -(T_MP[:3, :3].T @ self.geo.face_n)                     # the force's direction on the mount (P axes)
        gv = np.r_[np.cross(x_P - self.o_P, n_P), n_P]
        c_eff = 0.004 + float(gv @ self.Cw @ gv)
        f = -gap / c_eff
        F = f * n_P
        self.wrench_P = np.r_[F, np.cross(x_P - self.o_P, F)]
        self.truth_row = np.r_[F, x_P, -n_P, 0.0, 0.0, 2.0]               # kind 2: the column face

    # ---------------------------------------------------------------- sensors
    def hex_reading(self):
        Tf = self.flange_true()
        R_SP = (C.inv(self.T_ES) @ self.T_EP)[:3, :3]
        f_S = R_SP @ self.wrench_P[:3]
        t_S = R_SP @ self.wrench_P[3:] * 1e-3                          # N m
        R_SE = self.T_ES[:3, :3].T
        gE = Tf[:3, :3].T @ np.array([0.0, 0.0, -C.G * self.tr.payload_kg])
        f_S = f_S + R_SE @ gE
        t_S = t_S + R_SE @ np.cross(np.array(self.tr.payload_com_mm) * 1e-3, gE)
        w = self.tr.hex_sign * np.r_[self.Kf @ f_S, self.Kt @ t_S]
        return w + self.off + self.rng.normal(0.0, 1.0, 6) * np.array(HEX_NOISE)

    def board_reading(self):
        Tf = self.flange_true()
        R_PB = (Tf @ self.T_EP)[:3, :3].T
        g = R_PB @ np.array(self.tr.earth_uT, float)
        b = self.model.rest_uT(g_uT=g)
        if len(self.Fidx):
            m = np.einsum("ka,kamq->mq", self.F, self.mot[self.Fidx])
            b = b + self.model.motion_uT(m).reshape(5, 3)
        if np.any(self.shift):
            m = np.zeros((5, 6))
            m[:, :3] = self.shift * 1e3
            b = b + self.model.motion_uT(m).reshape(5, 3)
        T = self.T_amb + self.tr.warm_C * (1.0 - math.exp(-(self.t - self.t_start) / self.tr.warm_tau_s))
        gain = 1.0 + self.kappa * (T - C_T_REF)
        b = b * gain[:, None] + self.rng.normal(0.0, 1.0, (5, 3)) * self.Wm.NOISE_UT[None]
        return np.c_[T + self.rng.normal(0.0, 0.01, 5), b]

    def advance(self, dt):
        t_end = self.t + dt
        while self.t < t_end - 1e-12:
            h = min(0.002, t_end - self.t)
            self.step_robot(h)
            self.t += h
            self.off[:3] += self.rng.normal(0.0, self.tr.hex_drift_N_rt_s * math.sqrt(h), 3)
            if self.t >= self.ft_next:
                self.contact()
                w = self.hex_reading()
                self.streams.ft_add(self.t, w)
                self.n_ft = getattr(self, "n_ft", 0) + 1
                if self.n_ft % 5 == 0:                                     # 100 Hz is enough for the holds
                    self.streams.truth_add(self.t, np.r_[self.truth_row, C.T_to_ur(self.flange_true())])
                self.ft_next += 1.0 / 500.0
            if self.t >= self.board_next:
                self.contact()
                self.streams.board_add(self.t, self.board_reading())
                self.board_next += 1.0 / 140.0
            if self.t >= self.robot_next:
                Tn = fk(self.q)
                self.streams.robot_add(self.t, np.r_[self.t - self.t_start, self.q, C.T_to_ur(Tn), 0.0])
                self.robot_next += 1.0 / 125.0
            if self.tr.seat_shift_um > 0 and len(self.Fidx) == 0 and self.peak_N > 3.0:
                self.shift = self.shift + self.rng.normal(0.0, self.tr.seat_shift_um * 1e-3 / math.sqrt(3), 3)
                self.peak_N = 0.0
                self.cache = None


C_T_REF = 25.0


# ------------------------------------------------------------------------------------------- the backend
class SimRing(C.Ring):
    pass


class SimStreams:
    def __init__(self):
        self.ft = SimRing()
        self.board = SimRing()
        self.robot = SimRing()
        self.app = dict(ft=None, board=None, robot=None, truth=None)
        self.buf = dict(ft=[], board=[], robot=[], truth=[])

    def flush(self, name):
        b = self.buf[name]
        if b and self.app[name] is not None:
            if name == "board":
                rows = np.zeros(len(b), C.BoardStream.DTYPE)
                for k, (t, r) in enumerate(b):
                    rows[k] = (t, r.ravel())
                self.app[name].write(rows)
            else:
                self.app[name].write(b)
        self.buf[name] = []

    def ft_add(self, t, w):
        self.ft.add(t, w)
        self.buf["ft"].append([t, 0.0, 0.0, *w])
        if len(self.buf["ft"]) >= 500:
            self.flush("ft")

    def board_add(self, t, r):
        self.board.add(t, r.ravel())
        self.buf["board"].append((t, r))
        if len(self.buf["board"]) >= 200:
            self.flush("board")

    def truth_add(self, t, row):
        self.buf["truth"].append([t, *row])
        if len(self.buf["truth"]) >= 500:
            self.flush("truth")

    def robot_add(self, t, row):
        self.robot.add(t, row)
        self.buf["robot"].append([t, *row])
        if len(self.buf["robot"]) >= 200:
            self.flush("robot")


class _Stream:
    def __init__(self, ring):
        self.ring = ring
        self.connected = True


class SimRobot:
    def __init__(self, w: World):
        self.w = w

    def pose(self):
        return fk(self.w.q)

    def q(self):
        return self.w.q.copy()

    def _run(self):
        while self.w.seg is not None:
            self.w.advance(0.004)

    def moveL(self, T, v=0.02, a=0.2, asynchronous=False):
        self.w.start_segment(self.w.path_segment([T], v * 1e3, a * 1e3))
        if not asynchronous:
            self._run()
        return True

    def moveL_path(self, Ts, v=0.005, a=0.05, blend=0.0002, asynchronous=False):
        self.w.start_segment(self.w.path_segment(Ts, v * 1e3, a * 1e3))
        if not asynchronous:
            self._run()
        return True

    def moveJ(self, q, v=0.3, a=0.5, asynchronous=False):
        q = np.asarray(q, float)
        self.w.start_segment(dict(kind="J", q0=self.w.q.copy(), q1=q, L=float(np.abs(q - self.w.q).max()), vmax=v, amax=a))
        if not asynchronous:
            self._run()
        return True

    def stopL(self, a=0.5):
        self.w.stop_segment(a * 1e3)

    def steady(self):
        return self.w.seg is None

    def ik(self, T, qnear):
        q = ik(T, qnear)
        if q is None or np.any(np.abs(q) > 2 * math.pi):
            return None
        return q

    def stopped(self):
        return self.w.prot

    def teach(self, on):
        self.w.teach = bool(on)
        return True

    def set_payload(self, kg, cog_mm):
        return True

    def close(self):
        pass


class SimBackend:
    def __init__(self, plan, seed=0, truth=None):
        tr = Truth(seed=seed)
        if truth:
            d = json.loads(Path(truth).read_text()) if isinstance(truth, (str, Path)) else dict(truth)
            tr = Truth(**{**asdict(tr), **d})
        self.truth = tr
        self.w = World(plan, tr)
        self.streams = SimStreams()
        self.w.streams = self.streams
        start = plan["sim"]["start"]
        T_BP = self.w.T_BI @ self.w.geo.T_IP(start["p"], start["n"], 0.0, start["standoff"])
        T_BE = T_BP @ C.inv(self.w.geo.T_EM0 @ self.w.geo.T_MP0)
        q = ik(T_BE, np.array(plan["sim"]["q_seed"], float))
        if q is None:
            raise SystemExit("the simulator's start pose is out of reach")
        self.w.q = q
        self.robot = SimRobot(self.w)
        self.ft = _Stream(self.streams.ft)
        self.board = _Stream(self.streams.board)
        self.robot.ring = self.streams.robot
        self.w.advance(0.05)

    def attach(self, log):
        self.streams.app = dict(ft=log.appender("ft.bin"), board=log.appender("board.bin", C.BoardStream.DTYPE),
                                robot=log.appender("robot.bin"), truth=log.appender("sim_truth.bin"))
        log.save("sim_truth.json", dict(truth=asdict(self.truth), truth_bin="t, f (3, N, pad frame), centre of pressure (3, mm, P), "
                                        "normal (3), torsion (N mm), contact points, kind (0 none, 1 pad, 2 column face), true flange "
                                        "pose (6, m, rotation vector)", T_EM=self.w.T_EM, T_MP=self.w.T_MP, T_ES=self.w.T_ES,
                                        T_BI=self.w.T_BI, c_B=self.w.c_B, T_EP=self.w.T_EP, Kf=self.w.Kf, Kt=self.w.Kt,
                                        dh_true=self.w.dh_true))

    def now(self):
        return self.w.t

    def sleep(self, dt):
        self.w.advance(dt)

    def operator(self, prompt, kind=None):
        print(f"[sim operator] {prompt}", flush=True)
        if kind == "seat":
            e = self.truth.seat_err_mm
            d = np.array([self.w.rng.uniform(-e[0], e[0]), self.w.rng.normal(0, e[1]), self.w.rng.normal(0, e[2])])
            T_BM = self.w.T_BI @ C.Tm(None, d) @ self.w.geo.T_IM_home
            T_BE = T_BM @ C.inv(self.w.T_EM)
            q = ik(T_BE, self.w.q, self.w.dh_true)
            if q is None:
                raise SystemExit("the simulated operator cannot seat the socket")
            self.w.start_segment(dict(kind="J", q0=self.w.q.copy(), q1=q, L=float(np.abs(q - self.w.q).max()), vmax=0.2, amax=0.4))
            while self.w.seg is not None:
                self.w.advance(0.01)
            self.w.rep = np.zeros(3)
            self.w.advance(0.3)
        return ""

    def close(self):
        for k in ("ft", "board", "robot", "truth"):
            self.streams.flush(k)
