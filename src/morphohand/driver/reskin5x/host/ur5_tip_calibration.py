#!/usr/bin/env python3
"""Calibration session of the five-chip magnetic fingertip on the lab's UR5 (CB3) against its OptoForce HEX-E.

The robot presses the pad of DIP link b v4 (in UR5 mount v3) on a printed indenter screwed to plywood, the 5 mm sphere
or the 12.5 mm half rod, while three streams are logged with host time stamps: the HEX-E (UDP, polled at 500 Hz), the
board (USB serial, about 140 records/s) and the robot (RTDE: joints and flange pose, 125 Hz on the CB3). The plan, written
on the workstation by scripts/ur5_tip_calibration_plan.py, gives every target in the pad frame P of the estimator
(scripts/fingertip_wrench_model.py: x along the face's normal, y across the finger, z toward the tip end, mm) and the
indenter frame I (z up from the indenter's base, the sphere's centre at (0, 0, 41)). This program turns the targets into
flange poses through the registration it measures, stage by stage:

  preflight  stream rates and noise, payload, TCP set to zero (every pose here is the flange's, tool0)
  homing     the operator seats the mount's homing socket on the indenter's key in freedrive, three times: the
             indenter's pose in the robot's base, through the mount's CAD
  gravity    about 16 orientations 150 mm above the indenter: the HEX-E's axes, sign, gains and offsets, the
             payload's weight and centre of mass (the board's rest readings there give its chips' axes in Earth's field)
  mount      the sphere touches the mount column's flat palmar face at four points (the onset of the force)
  pad        the sphere touches the pad along the normal at about 40 points over the face, the edges and the flanks
  register   one least-squares fit of the mount on the flange, the pad on the mount and the sphere's centre
  pivot      the face centre pressed to 1.5 N with four 0.3 mm offsets and from four turns about its normal: the board's
             pattern shows where each press landed, the fit is repeated with them
  grid       the calibration presses: each point from 1.5 mm off to force levels up to 5 N, 0.6 s per level, a rest
             before and after; the reference press (face centre to 0.5, 1, 2 N) every 15 presses
  shear      in contact, turns about the sphere's centre: sliding at constant indentation, then gross slip
  roll       turns about the contact point: the contact rolls over the pad
  twist      turns about the normal through the contact: torsion
  final      the reference press and six gravity orientations again (drift of the HEX-E and of the seat)

Every action is logged in <session>/events.jsonl as it starts and ends (fsync); the streams go to ft.bin, board.bin and
robot.bin. A session started again with --resume <session> skips the actions that ended ok and reloads the
registration. --dry 15 shifts every target 15 mm away from the indenter along its approach (no contact: reach, cables).
--step asks before each action. Ctrl-C stops the robot (stopL) and closes the files.

    python ur5_tip_calibration.py --plan plan.json --robot 192.168.0.4 --ft 192.168.0.5
    python ur5_tip_calibration.py --plan plan.json --sim                 the same session against the simulator
    ... --stages preflight,homing,gravity     a subset, in the order given
Files needed on the robot's computer: this file, reskin5x_plot.py (the board's reader, same folder) and the plan; Python
with numpy, pyserial and ur_rtde (pip install ur_rtde).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PLAN_VERSION = 1
G = 9.80665
EZ = np.array([0.0, 0.0, 1.0])


# ------------------------------------------------------------------------------------------- rigid motions (mm, rad)
def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def exp_so3(w):
    w = np.asarray(w, float)
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    K = skew(w / th)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * K @ K


def log_so3(R):
    c = min(1.0, max(-1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-9:
        return v / 2.0
    if math.pi - th < 1e-6:
        A = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(A)))
        k = A[:, i] / math.sqrt(max(A[i, i], 1e-12))
        return k * th
    return th / (2.0 * math.sin(th)) * v


def Tm(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def inv(T):
    R, t = T[:3, :3], T[:3, 3]
    return Tm(R.T, -R.T @ t)


def apply(T, p):
    return np.asarray(p, float) @ T[:3, :3].T + T[:3, 3]


def ur_to_T(pose):
    """UR pose [x, y, z (m), rotation vector] to a 4 x 4 transform in mm."""
    return Tm(exp_so3(pose[3:6]), np.asarray(pose[:3], float) * 1e3)


def T_to_ur(T):
    return [float(v) for v in T[:3, 3] * 1e-3] + [float(v) for v in log_so3(T[:3, :3])]


def rot(axis, deg):
    a = np.asarray(axis, float)
    return exp_so3(a / np.linalg.norm(a) * math.radians(deg))


def rot_from_to(a, b):
    """The least rotation taking direction a to direction b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    s, c = float(np.linalg.norm(v)), float(a @ b)
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        return exp_so3(p / np.linalg.norm(p) * math.pi)
    return exp_so3(v / s * math.atan2(s, c))


def perturb(T, d):
    """T followed by the small motion d = [rotation vector (rad), translation (mm)] in T's own frame."""
    return T @ Tm(exp_so3(d[:3]), np.asarray(d[3:6], float))


def turn_about(T, c, axis, deg):
    """Pose T (world) turned by deg about the line through point c (world) along axis (world)."""
    R = rot(axis, deg)
    return Tm(R, np.asarray(c, float) - R @ np.asarray(c, float)) @ T


def pose_error(Ta, Tb):
    """(mm, deg) between two poses."""
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), math.degrees(float(np.linalg.norm(log_so3(Ta[:3, :3].T @ Tb[:3, :3]))))


def mean_pose(Ts):
    Ts = list(Ts)
    R0 = Ts[0][:3, :3]
    w = np.mean([log_so3(R0.T @ T[:3, :3]) for T in Ts], axis=0)
    return Tm(R0 @ exp_so3(w), np.mean([T[:3, 3] for T in Ts], axis=0))


def jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


# ------------------------------------------------------------------------------------------- the plan's geometry
def closest_on_triangles(p, A, B, C):
    """Closest point to p on each triangle (A, B, C rows): (n, 3). Ericson, Real-Time Collision Detection 5.1.5."""
    ab, ac, ap = B - A, C - A, p - A
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp = p - B
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    cp = p - C
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-18, 1e-18, den)
    v, w = vb / den, vc / den
    out = A + ab * v[:, None] + ac * w[:, None]                     # inside the face
    with np.errstate(divide="ignore", invalid="ignore"):
        m_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        out = np.where(m_ab[:, None], A + ab * (d1 / (d1 - d3))[:, None], out)
        m_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        out = np.where(m_ac[:, None], A + ac * (d2 / (d2 - d6))[:, None], out)
        m_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        out = np.where(m_bc[:, None], B + (C - B) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))[:, None], out)
    out = np.where(((d1 <= 0) & (d2 <= 0))[:, None], A, out)
    out = np.where(((d3 >= 0) & (d4 <= d3))[:, None], B, out)
    out = np.where(((d6 >= 0) & (d5 <= d6))[:, None], C, out)
    return out


class Geometry:
    """Frames (mm): B the robot's base, E its flange (tool0), S the HEX-E, M the mount (origin on the HEX-E's tool face,
    z along the tool axis toward the tip, x toward the palmar side and the dowel), P the pad (the estimator's tip frame),
    I the indenter (z up from its base). Nominal transforms come from the plan; Registration holds the measured ones."""

    def __init__(self, plan):
        g = plan["geometry"]
        self.T_EM0 = np.array(g["T_EM"], float)
        self.T_MP0 = np.array(g["T_MP"], float)
        self.T_ES0 = np.array(g["T_ES"], float)
        self.T_IM_home = np.array(g["T_IM_home"], float)
        ind = g["indenter"]
        self.kind = ind["kind"]
        self.c_I = np.array(ind["centre"], float)          # sphere: centre; rod: a point on its axis
        self.R = float(ind["radius"])
        self.rod_axis = np.array(ind.get("axis", [0.0, 1.0, 0.0]), float)
        s = g["surface"]
        self.pts = np.array(s["pts"], float)
        self.nrm = np.array(s["normals"], float)
        self.tris = np.array(s["tris"], int)
        A, B, C = (self.pts[self.tris[:, k]] for k in range(3))
        n = np.cross(B - A, C - A)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        vn = self.nrm[self.tris].sum(axis=1)
        n *= np.sign(np.einsum("ij,ij->i", n, vn))[:, None]       # outward, as the vertex normals
        self.tA, self.tB, self.tC, self.tN = A, B, C, n
        self.face = g["mount_face"]                                 # plane of the column's palmar face, in M
        self.face_n = np.array(self.face["normal"], float)
        self.face_d = float(np.array(self.face["point"], float) @ self.face_n)
        self.R_ref = self.T_IM_home[:3, :3]                         # pad axes to I at homing: x_P -> -z_I, z_P -> x_I

    def surface_distance(self, q):
        """Signed distance (mm, outside positive), closest point and outward normal of the pad's free surface to q (P)."""
        cp = closest_on_triangles(np.asarray(q, float), self.tA, self.tB, self.tC)
        d = np.linalg.norm(cp - q, axis=1)
        k = int(np.argmin(d))
        s = 1.0 if (q - cp[k]) @ self.tN[k] >= 0 else -1.0
        return s * float(d[k]), cp[k], self.tN[k]

    def contact_point_I(self, standoff, y_rod=0.0):
        """Where the target point sits in I: standoff above the sphere's top (or the rod's top line)."""
        if self.kind == "sphere":
            return self.c_I + (self.R + standoff) * EZ
        return self.c_I + y_rod * self.rod_axis + (self.R + standoff) * EZ

    def R_IP(self, n_P, twist_deg):
        """Pad axes in I with the pad's normal n_P pointing down (-z_I), turned by twist about the vertical."""
        return rot(EZ, twist_deg) @ self.R_ref @ rot_from_to(n_P, [1.0, 0.0, 0.0])

    def T_IP(self, p_P, n_P, twist_deg, standoff, offset_I=(0.0, 0.0), y_rod=0.0):
        R = self.R_IP(n_P, twist_deg)
        c = self.contact_point_I(standoff, y_rod) - np.array([offset_I[0], offset_I[1], 0.0])
        return Tm(R, c - R @ np.asarray(p_P, float))


class Registration:
    """The measured transforms and the HEX-E's gravity model; saved as registration.json in the session."""

    def __init__(self, geo: Geometry):
        self.T_EM, self.T_MP, self.T_ES = geo.T_EM0.copy(), geo.T_MP0.copy(), geo.T_ES0.copy()
        self.T_BI = None                    # indenter in the base (homing; position refined by the fit)
        self.T_BE_home = None
        self.q_home = None                  # the IK branch of the homing pose: joint moves solve nearest it
        self.grav = None                    # dict A, of, Bt, ot (f_S = A g_E + of, tau_S = Bt g_E + ot; g_E in units of g)
        self.K = np.eye(3)                  # f_S = K f_E: the HEX-E's force axes, sign and gains on the flange
        self.Kt = np.eye(3)                 # torques likewise (the rotation and sign of K)
        self.notes = {}

    def T_EP(self):
        return self.T_EM @ self.T_MP

    def to_json(self):
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items() if k != "grav"}
        d["grav"] = None if self.grav is None else {k: np.asarray(v).tolist() for k, v in self.grav.items()}
        return d

    def load(self, d):
        for k, v in d.items():
            if k == "grav":
                self.grav = None if v is None else {kk: np.array(vv, float) for kk, vv in v.items()}
            elif k == "notes":
                self.notes = v
            elif v is not None:
                setattr(self, k, np.array(v, float))
        return self


# ------------------------------------------------------------------------------------------- fits
def fit_gravity(R_BE, f_S, t_S, g_B=(0.0, 0.0, -1.0), t=None):
    """Payload sweep: f_S = A g_E + of and tau_S = Bt g_E + ot, g_E the unit gravity in the flange frame (R_BE^T g_B).
    A = m K with K the HEX-E's map from flange-frame force to its reading (axes, sign, gains, crosstalk); m is taken
    as A's mean singular value (the HEX-E's factory gain kept). Bt = m Kt [c]x gives the centre of mass c (from the
    sensor's moment origin, flange axes) when Kt is the rotation and sign of K.

    With t (s, each record's time) the HEX-E's drift during the sweep is fitted as well: f_S and tau_S gain a term
    v (t - t_mean), v_f in N/min and v_t in N m/min, and of and ot are the offsets at t_mean. With the arm still, the
    lab's untared Fz crept 0.08 N/min on 2026-10-22; added to a simulated sweep, that drift put A 3.3 % off in a 4 min
    sweep at 30 % speed without the term, and 0.3 % off with it (QUEUE.md R6)."""
    gE = np.einsum("kji,j->ki", np.asarray(R_BE), np.asarray(g_B, float))          # (k, 3) = R^T g
    X = np.c_[gE, np.ones(len(gE))]
    if t is not None:
        t = np.asarray(t, float)
        t_mean = float(t.mean())
        X = np.c_[X, (t - t_mean) / 60.0]
    sol_f, *_ = np.linalg.lstsq(X, np.asarray(f_S), rcond=None)
    sol_t, *_ = np.linalg.lstsq(X, np.asarray(t_S), rcond=None)
    A, of = sol_f[:3].T, sol_f[3]
    Bt, ot = sol_t[:3].T, sol_t[3]
    U, s, Vt = np.linalg.svd(A)
    Q = U @ Vt                                                    # nearest orthogonal (polar factor)
    sign = float(np.sign(np.linalg.det(Q)))
    R_SE = sign * Q                                               # a proper rotation
    m_N = float(np.mean(s))                                       # weight in newtons
    K = A / m_N
    Kt = sign * R_SE
    Cx = np.linalg.solve(m_N * Kt, Bt)                           # [c]x in m (torques N m, weight N)
    c_m = np.array([Cx[2, 1] - Cx[1, 2], Cx[0, 2] - Cx[2, 0], Cx[1, 0] - Cx[0, 1]]) / 2.0
    res_f = np.asarray(f_S) - X @ sol_f
    res_t = np.asarray(t_S) - X @ sol_t
    ang = math.degrees(float(np.linalg.norm(log_so3(R_SE))))
    z_turn = math.degrees(math.atan2(R_SE[1, 0], R_SE[0, 0]))
    out = dict(A=A, of=of, Bt=Bt, ot=ot, K=K, Kt=Kt, R_SE=R_SE, sign=sign, weight_N=m_N, mass_kg=m_N / G,
               com_mm=c_m * 1e3, sv=s, rms_f_N=float(np.sqrt(np.mean(res_f ** 2))),
               rms_t_Nm=float(np.sqrt(np.mean(res_t ** 2))), turn_deg=ang, z_turn_deg=z_turn,
               skew_of_K=float(np.linalg.norm(K - K.T) / np.linalg.norm(K)), n=len(gE))
    if t is not None:
        out.update(v_f=sol_f[4], v_t=sol_t[4], t_mean=t_mean)
    return out


def fit_onset(s, y, y_max, p_set=(1.0, 1.25, 1.5, 1.75, 2.0), step=0.002, win=25, span_mm=0.5):
    """Onset s0 (mm along the approach) of a load y (N, time order) that rises as A (s - s0)^p past it. The rise is found
    on a moving average over win samples (the HEX-E's noise, 0.035 N a sample laterally, crosses small thresholds by
    itself): samples after the average first passes y_max are left out, and s0 is searched over the span_mm before the
    last sample where the average lies under a quarter of y_max. Returns (s0, p, rms, n_contact) or None."""
    s, y = np.asarray(s, float), np.asarray(y, float)
    if len(s) < 2 * win:
        return None
    ys = np.convolve(y, np.ones(win) / win, mode="same")
    above = np.nonzero(ys > y_max)[0]
    end = int(above[0]) if len(above) else len(y)
    s, y, ys = s[:end], y[:end], ys[:end]
    if len(s) < 2 * win or ys[win:].max() < 0.3 * y_max:
        return None
    below = np.nonzero(ys[:max(1, len(ys) - win // 2)] < 0.25 * y_max)[0]
    s_hi = float(s[below[-1]]) if len(below) else float(s[0])
    keep = s >= s_hi - span_mm - 0.1
    s, y = s[keep], y[keep]
    best = None
    for p in p_set:
        for s0 in np.arange(s_hi - span_mm, s_hi + step, step):
            x = np.clip(s - s0, 0.0, None) ** p
            X = np.c_[x, np.ones_like(x)]
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            if coef[0] <= 0:
                continue
            r = float(np.sqrt(np.mean((X @ coef - y) ** 2)))
            if best is None or r < best[2]:
                best = (float(s0), float(p), r, int((s > s0).sum()))
    return best


def lm(fun, x0, iters=40, h=1e-6, lam=1e-3):
    """Levenberg-Marquardt with a forward-difference Jacobian; fun returns the residual vector."""
    x = np.array(x0, float)
    r = fun(x)
    cost = float(r @ r)
    for _ in range(iters):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            dx = np.zeros_like(x)
            dx[j] = h * max(1.0, abs(x[j]))
            J[:, j] = (fun(x + dx) - r) / dx[j]
        Hm = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(8):
            step = np.linalg.solve(Hm + lam * np.diag(np.diag(Hm) + 1e-12), -g)
            r2 = fun(x + step)
            c2 = float(r2 @ r2)
            if c2 < cost:
                x, r, cost, lam = x + step, r2, c2, max(lam / 3.0, 1e-9)
                improved = True
                break
            lam *= 4.0
        if not improved or np.linalg.norm(step) < 1e-9:
            break
    J = np.empty((len(r), len(x)))
    for j in range(len(x)):
        dx = np.zeros_like(x)
        dx[j] = h * max(1.0, abs(x[j]))
        J[:, j] = (fun(x + dx) - r) / dx[j]
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((len(x), len(x)), np.nan)
    return x, r, cov


REG_SIGMA = dict(pad_mm=0.03, face_mm=0.02, pivot_mm=0.10, home_mm=(0.25, 0.10, 0.12),
                 em_rot_deg=(0.5, 0.5, 2.0), em_t_mm=(0.4, 0.4, 0.6), mp_rot_deg=(0.6, 0.6, 0.6), mp_t_mm=(0.3, 0.3, 0.3))


def fit_registration(geo: Geometry, reg: Registration, touches, pivots=(), sig=REG_SIGMA):
    """The mount on the flange (T_EM), the pad on the mount (T_MP) and the sphere's centre in the base, from the pad and
    mount-face touches (flange pose at the onset of contact), the homing pose and the CAD; pivots add where the board
    placed the pivot presses relative to the reference press. Returns a dict with the new transforms and residuals."""
    T_BI0 = reg.T_BI
    c0 = apply(T_BI0, geo.c_I)
    R_BI = T_BI0[:3, :3]
    T_EM0, T_MP0 = reg.T_EM.copy(), reg.T_MP.copy()
    T_MI_home = inv(geo.T_IM_home)
    home = reg.T_BE_home
    pad = [t for t in touches if t["kind"] == "pad"]
    face = [t for t in touches if t["kind"] == "mount"]
    sd_em = np.r_[np.radians(sig["em_rot_deg"]), sig["em_t_mm"]]
    sd_mp = np.r_[np.radians(sig["mp_rot_deg"]), sig["mp_t_mm"]]

    def unpack(x):
        T_EM = perturb(T_EM0, x[0:6])
        T_MP = perturb(T_MP0, x[6:12])
        return T_EM, T_MP, c0 + x[12:15]

    def resid(x):
        T_EM, T_MP, c = unpack(x)
        T_EP = T_EM @ T_MP
        r = []
        for t in pad:
            q = apply(inv(np.array(t["T_BE"]) @ T_EP), c)
            r.append((geo.surface_distance(q)[0] - geo.R) / sig["pad_mm"])
        for t in face:
            q = apply(inv(np.array(t["T_BE"]) @ T_EM), c)
            r.append((q @ geo.face_n - geo.face_d - geo.R) / sig["face_mm"])
        if home is not None:
            ch = apply(home @ T_EM @ T_MI_home, geo.c_I)
            r.extend(list((R_BI.T @ (c - ch)) / np.array(sig["home_mm"])))
        for pv in pivots:                                         # tangential position of the press relative to the reference
            q0 = apply(inv(np.array(pv["T_BE_ref"]) @ T_EP), c)
            q1 = apply(inv(np.array(pv["T_BE"]) @ T_EP), c)
            t1, t2 = np.array(pv["t1"]), np.array(pv["t2"])
            pred = np.array([(q1 - q0) @ t1, (q1 - q0) @ t2])
            r.extend(list((pred - np.array(pv["dp_board"])) / sig["pivot_mm"]))
        r.extend(list(x[0:6] / sd_em))
        r.extend(list(x[6:12] / sd_mp))
        return np.array(r)

    x, r, cov = lm(resid, np.zeros(15))
    T_EM, T_MP, c = unpack(x)
    n_pad, n_face = len(pad), len(face)
    out = dict(T_EM=T_EM, T_MP=T_MP, c_B=c, x=x, sd=np.sqrt(np.clip(np.diag(cov), 0, None)),
               rms_pad_mm=float(np.sqrt(np.mean((r[:n_pad] * sig["pad_mm"]) ** 2))) if n_pad else float("nan"),
               rms_face_mm=float(np.sqrt(np.mean((r[n_pad:n_pad + n_face] * sig["face_mm"]) ** 2))) if n_face else float("nan"),
               n_pad=n_pad, n_face=n_face, n_pivot=len(pivots), chi2=float(r @ r), dof=int(len(r) - 15))
    T_EP0, T_EP1 = T_EM0 @ T_MP0, T_EM @ T_MP
    out["tcp_change_mm_deg"] = pose_error(T_EP0, T_EP1)
    out["centre_change_mm"] = float(np.linalg.norm(c - c0))
    return out


# ------------------------------------------------------------------------------------------- streams (real hardware)
class Ring:
    """Thread-safe buffer of the last keep_s seconds of (t, row) samples."""

    def __init__(self, keep_s=180.0):
        self.keep, self.lock = keep_s, threading.Lock()
        self.t, self.v = deque(), deque()

    def add(self, t, v):
        with self.lock:
            self.t.append(t)
            self.v.append(v)
            while self.t and t - self.t[0] > self.keep:
                self.t.popleft()
                self.v.popleft()

    def since(self, t0, t1=None):
        with self.lock:
            t = np.array(self.t)
            v = np.array(self.v) if self.v else np.zeros((0, 1))
        if len(t) == 0:
            return t, v
        m = t >= t0
        if t1 is not None:
            m &= t <= t1
        return t[m], v[m]

    def latest(self):
        with self.lock:
            return (self.t[-1], self.v[-1]) if self.t else (None, None)


class Appender:
    """Binary file of float64 rows, flushed at most every 0.5 s and fsynced every 5 s by a thread of its own.

    The fsync used to run in the writer's thread, which for ft.bin is the HEX-E's reader: on the lab computer's SSD it
    blocked for 0.1 to 0.4 s, and the reader's pauses of 2026-10-22 (0.36 s at 21:14:59, then 0.08 to 0.15 s every 0.6 s
    from 21:17:45) aborted three touch-offs with no HEX-E sample for 100 ms while the sensor sampled on."""

    def __init__(self, path, dtype=None):
        self.f = open(path, "ab")
        self.dtype = dtype
        self.last = 0.0
        self.lock = threading.Lock()
        self.closed = False
        threading.Thread(target=self._sync, daemon=True).start()

    def _sync(self):
        while not self.closed:
            time.sleep(5.0)
            try:
                os.fsync(self.f.fileno())
            except (OSError, ValueError):                        # closed meanwhile
                return

    def write(self, rows):
        with self.lock:
            if self.dtype is None:
                self.f.write(np.asarray(rows, "<f8").tobytes())
            else:
                self.f.write(np.asarray(rows, self.dtype).tobytes())
            now = time.time()
            if now - self.last > 0.5:
                self.f.flush()
                self.last = now

    def close(self):
        with self.lock:
            self.closed = True
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()


class OptoForceStream(threading.Thread):
    """The HEX-E through its Ethernet DAQ: UDP to port 49152, the DAQ's datagrams of magpie_control (ft_sensor.py,
    magpie_force_control/src/ft_sensor.cpp): header 0x1234, command 0x0082 speed (1000 / value Hz: 2 is 500 Hz), 0x0081
    low-pass filter (0 none; 2, 3, 4 cut at 150, 50, 15 Hz), 0x0042 bias (1 tares), 0x0002 send n samples; answers of 3
    uint32 (sequence numbers, status) and 6
    int32, forces / 1e4 N, torques / 1e5 N m. Polled one sample at a time (the lab's verified way, as in magpie_control's
    reader since commit 806b94a of 2026-03-06); stream=True asks for continuous sending instead. Rows: t_host,
    ft_sequence, status, Fx, Fy, Fz, Tx, Ty, Tz.

    A datagram that waits in the socket is stale: an answer that arrived after its read timed out, or one the DAQ sent
    on its own because a program quit while it streamed. Read later, it would pass for the answer to a newer request,
    and the lag would grow by one sample each time (the 20 ms reader of 2024-11 in magpie_control piled them up this
    way). So configure() stops the DAQ and empties the socket, and each polled request first empties it again;
    n_late counts the datagrams dropped.

    A gap of over 0.5 s between samples goes into `gaps` as (host time before, after, sequence before, after, change of
    Fz in N). On 2026-10-22 the sensor dropped out twice while joint 6 turned; after the first gap its sample counter
    had restarted (13702, then 102) and its tare was lost (Fz from 0.3 to 645.5 N). The session treats an action with a
    gap as aborted.

    The DAQ hands out single samples, so a slower poll alone leaves the noise as it is; its low-pass lowers it. Polled at
    100 Hz on 2026-10-22 (lab computer, arm off), Fz's noise per sample was 0.53 N with no filter (as at 250 and 500 Hz),
    0.35, 0.21 and 0.13 N with filters 2, 3 and 4 (owner: "100-200 is a better range. at 500 things are really noise")."""

    SPEED = {500: 2, 250: 4, 100: 10, 50: 20}

    def __init__(self, ip, port=49152, rate=100, appender=None, stream=False, bias=True, filt=3):
        super().__init__(daemon=True)
        self.addr, self.rate, self.app, self.stream_mode, self.bias = (ip, port), rate, appender, stream, bias
        self.filt = int(filt)
        self.ring = Ring()
        self.stop, self.n, self.n_timeout, self.n_repeat, self.n_late = False, 0, 0, 0, 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.05)
        self.sock.connect(self.addr)
        self.last_seq, self.last_t, self.last_v, self.last_seq_kept, self.gaps = None, None, None, None, []

    def _cmd(self, cid, value, wait=0.02):
        self.sock.send(struct.pack("!HHI", 0x1234, cid, value))
        time.sleep(wait)

    def drain(self):
        n = 0
        self.sock.setblocking(False)
        try:
            while True:
                self.sock.recv(64)
                n += 1
        except (BlockingIOError, InterruptedError):
            pass
        except OSError:                                          # ICMP port unreachable while nothing answers
            pass
        finally:
            self.sock.settimeout(0.05)
        return n

    def configure(self):
        self._cmd(0x0000, 0)
        self.n_late += self.drain()
        self._cmd(0x0082, self.SPEED[self.rate])
        self._cmd(0x0081, self.filt)
        if self.bias:
            self._cmd(0x0042, 1)
        if self.stream_mode:
            self._cmd(0x0002, 0)

    def tare(self):
        self._cmd(0x0042, 1, wait=0.1)

    def run(self):
        period = 1.0 / self.rate
        req = struct.pack("!HHI", 0x1234, 0x0002, 1)
        nxt, rows = time.time(), []
        while not self.stop:
            if not self.stream_mode:
                self.n_late += self.drain()
                self.sock.send(req)
            try:
                data = self.sock.recv(64)
            except socket.timeout:
                self.n_timeout += 1
                continue
            except OSError:                                      # ICMP port unreachable: the DAQ is not up yet
                self.n_timeout += 1
                time.sleep(0.05)
                continue
            t = time.time()
            if len(data) != 36:
                continue
            u = struct.unpack("!3I6i", data)
            seq = u[1]
            if seq == self.last_seq:
                self.n_repeat += 1
            else:
                self.last_seq = seq
                v = np.array([u[3] / 1e4, u[4] / 1e4, u[5] / 1e4, u[6] / 1e5, u[7] / 1e5, u[8] / 1e5])
                if self.last_t is not None and t - self.last_t > 0.5:
                    self.gaps.append((self.last_t, t, self.last_seq_kept, seq, float(v[2] - self.last_v[2])))
                self.last_t, self.last_v, self.last_seq_kept = t, v, seq
                self.ring.add(t, v)
                self.n += 1
                rows.append([t, seq, u[2], *v])
                if self.app is not None and len(rows) >= 50:
                    self.app.write(rows)
                    rows = []
            if not self.stream_mode:
                nxt += period
                dt = nxt - time.time()
                if dt > 0:
                    time.sleep(dt)
                elif dt < -0.05:
                    nxt = time.time()
        if self.app is not None and rows:
            self.app.write(rows)
        try:
            self._cmd(0x0000, 0, wait=0.0)
        except OSError:
            pass


class BoardStream(threading.Thread):
    """The board through reskin5x_plot.Reader: its records (host time, 5 chips x [T, x, y, z]) are drained every 0.2 s
    into board.bin (the contact viewer's stream format: t <f8, 20 x <f4) and a ring of the last 3 minutes."""

    DTYPE = np.dtype([("t", "<f8"), ("r", "<f4", (20,))])

    def __init__(self, port, appender=None):
        super().__init__(daemon=True)
        sys.path.insert(0, str(HERE))
        import reskin5x_plot as RP
        self.reader = RP.Reader(port, keep=5.0)
        self.app, self.ring, self.stop, self.n = appender, Ring(), False, 0

    def run(self):
        self.reader.start()
        while not self.stop:
            time.sleep(0.2)
            with self.reader.lock:
                recs, self.reader.all = self.reader.all, []
            if not recs:
                continue
            rows = np.zeros(len(recs), self.DTYPE)
            for k, (t, r) in enumerate(recs):
                rows[k] = (t, r.ravel())
                self.ring.add(t, r.ravel())
            self.n += len(recs)
            if self.app is not None:
                self.app.write(rows)
        self.reader.stop = True

    @property
    def connected(self):
        return self.reader.connected


class URRobot:
    """ur_rtde on the CB3: flange poses (the controller's TCP set to zero), mm and 4 x 4 here, m and rotation vectors
    there. A thread logs t_host, t_controller, q (6), flange pose (6, m and rad), safety mode at 125 Hz."""

    def __init__(self, ip, appender=None, rate=125.0):
        import rtde_control
        import rtde_receive
        self.c = rtde_control.RTDEControlInterface(ip)
        self.r = rtde_receive.RTDEReceiveInterface(ip, rate)
        self.c.setTcp([0.0] * 6)
        self.app, self.ring, self.stop, self.rate = appender, Ring(), False, rate
        self.th = threading.Thread(target=self._log, daemon=True)
        self.th.start()

    def _log(self):
        rows, per = [], 1.0 / self.rate
        while not self.stop:
            t0 = time.time()
            try:
                q, p = self.r.getActualQ(), self.r.getActualTCPPose()
                row = [t0, self.r.getTimestamp(), *q, *p, float(self.r.getSafetyMode())]
            except Exception:                                    # noqa: BLE001  (a dropped RTDE packet)
                time.sleep(per)
                continue
            self.ring.add(t0, np.array(row[1:]))
            rows.append(row)
            if self.app is not None and len(rows) >= 25:
                self.app.write(rows)
                rows = []
            dt = per - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
        if self.app is not None and rows:
            self.app.write(rows)

    def pose(self):
        return ur_to_T(self.r.getActualTCPPose())

    def q(self):
        return np.array(self.r.getActualQ())

    def moveL(self, T, v=0.02, a=0.2, asynchronous=False):
        return self.c.moveL(T_to_ur(T), v, a, asynchronous)

    def moveL_path(self, Ts, v=0.005, a=0.05, blend=0.0002, asynchronous=False):
        path = [T_to_ur(T) + [v, a, blend] for T in Ts]
        path[-1][-1] = 0.0
        return self.c.moveL(path, asynchronous)

    def moveJ(self, q, v=0.3, a=0.5, asynchronous=False):
        return self.c.moveJ(list(map(float, q)), v, a, asynchronous)

    def stopL(self, a=0.5):
        self.c.stopL(a)

    def steady(self):
        return self.c.isSteady()

    def ik(self, T, qnear):
        x = T_to_ur(T)
        try:
            if not self.c.getInverseKinematicsHasSolution(x, list(qnear)):
                return None
        except Exception:                                        # noqa: BLE001  (older controllers)
            pass
        q = self.c.getInverseKinematics(x, list(qnear))
        if not q:
            return None
        try:                                                     # the controller's IK can come back off the target:
            e = pose_error(ur_to_T(self.c.getForwardKinematics(list(q), [0.0] * 6)), T)   # grav-03 of 2026-10-22, 22.8 deg
        except Exception:                                        # noqa: BLE001  (older controllers)
            e = (0.0, 0.0)
        if e[0] > 1.0 or e[1] > 0.5:
            return None
        return np.array(q)

    def stopped(self):
        return bool(self.r.isProtectiveStopped() or self.r.isEmergencyStopped())

    def teach(self, on):
        return self.c.teachMode() if on else self.c.endTeachMode()

    def set_payload(self, kg, cog_mm):
        return self.c.setPayload(float(kg), [float(v) * 1e-3 for v in cog_mm])

    def close(self):
        self.stop = True
        try:
            self.c.stopScript()
            self.c.disconnect()
            self.r.disconnect()
        except Exception:                                        # noqa: BLE001
            pass


class RealBackend:
    def __init__(self, args, log):
        self.ft = OptoForceStream(args.ft, rate=args.ft_rate, appender=log.appender("ft.bin"), stream=args.ft_stream,
                                  filt=args.ft_filter)
        self.ft.configure()
        self.ft.start()
        self.board = BoardStream(args.board, appender=log.appender("board.bin", BoardStream.DTYPE))
        self.board.start()
        self.robot = URRobot(args.robot, appender=log.appender("robot.bin"))

    def now(self):
        return time.time()

    def sleep(self, dt):
        time.sleep(dt)

    def operator(self, prompt, kind=None):
        return input(f"\n>>> {prompt} [Enter] ")

    def close(self):
        self.ft.stop = True
        self.board.stop = True
        self.robot.close()


# ------------------------------------------------------------------------------------------- session log
class SessionLog:
    def __init__(self, root, name, resume=None, clock=time.time):
        if resume:
            self.dir = Path(resume)
        else:
            stamp = time.strftime("%Y%m%d")
            self.dir = Path(root) / f"{stamp}-tip_calibration" / f"{time.strftime('%H%M%S')}-{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.ev = open(self.dir / "events.jsonl", "a")
        self.apps = []

    def appender(self, name, dtype=None):
        a = Appender(self.dir / name, dtype)
        self.apps.append(a)
        return a

    def event(self, **kw):
        kw = dict(t=self.clock(), **kw)
        self.ev.write(json.dumps(kw, default=jsonable) + "\n")
        self.ev.flush()
        os.fsync(self.ev.fileno())

    def done(self):
        out = set()
        p = self.dir / "events.jsonl"
        if p.exists():
            for line in open(p):
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("event") == "end" and e.get("status") == "ok":
                    out.add(e["id"])
        return out

    def records(self, kind):
        out = []
        for line in open(self.dir / "events.jsonl"):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == kind:
                out.append(e)
        return out

    def save(self, name, obj):
        p = self.dir / name
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, indent=1, default=jsonable))
        os.replace(tmp, p)

    def close(self):
        for a in self.apps:
            a.close()
        self.ev.close()


# ------------------------------------------------------------------------------------------- the session
class Abort(Exception):
    pass


class Session:
    def __init__(self, plan, be, log, args):
        self.plan, self.be, self.log, self.args = plan, be, log, args
        self.geo = Geometry(plan)
        self.reg = Registration(self.geo)
        self.safety = dict(plan.get("safety", {}))
        self.wrist_window = tuple(float(v) for v in str(getattr(args, "wrist_window", "30,100")).split(","))
        self.safety.setdefault("abort_N", 10.0)
        self.safety.setdefault("abort_Nm", 1.5)
        self.safety.setdefault("box_mm", 400.0)
        self.touches, self.pivots = [], []
        self.n_abort = 0
        self.dry = float(args.dry or 0.0)
        self.done = log.done() if args.resume else set()
        reg_path = log.dir / "registration.json"
        if args.resume and reg_path.exists():
            self.reg.load(json.loads(reg_path.read_text()))
            for name, attr in (("touches.json", "touches"), ("pivots.json", "pivots")):
                if (log.dir / name).exists():
                    setattr(self, attr, json.loads((log.dir / name).read_text()))
            print(f"resumed {log.dir}: {len(self.done)} actions done, {len(self.touches)} touches")

    # ---------------------------------------------------------------- helpers
    def now(self):
        return self.be.now()

    def sleep(self, dt):
        self.be.sleep(dt)

    def save_reg(self):
        self.log.save("registration.json", self.reg.to_json())
        self.log.save("touches.json", self.touches)

    def ft_since(self, t0, t1=None):
        t, v = self.be.ft.ring.since(t0, t1)
        return t, v.reshape(-1, 6) if len(t) else np.zeros((0, 6))

    def board_since(self, t0, t1=None):
        t, v = self.be.board.ring.since(t0, t1)
        return t, v.reshape(-1, 20) if len(t) else np.zeros((0, 20))

    def robot_since(self, t0, t1=None):
        t, v = self.be.robot.ring.since(t0, t1)
        return t, v

    def wait_steady(self, timeout=30.0):
        t0 = self.now()
        while not self.be.robot.steady():
            self.check_ft()
            self.sleep(0.004)
            if self.now() - t0 > timeout:
                raise Abort("the robot did not come to rest")

    def wrist_ok(self, q):
        lo, hi = self.wrist_window
        return lo <= math.degrees(float(q[4])) <= hi

    def check_wrist(self, q=None, target=False):
        """Abort when joint 5 leaves --wrist-window (deg): near 0 and 180 deg the wrist is singular, and past about 90
        deg the HEX-E's cable port can reach the forearm (2026-10-22: pressed at 173.5 deg, the owner stopped it)."""
        q = self.be.robot.q() if q is None else q
        if not self.wrist_ok(q):
            if not target:
                self.be.robot.stopL(3.0)
            raise Abort(f"joint 5 {'target' if target else 'at'} {math.degrees(float(q[4])):.1f} deg, outside "
                        f"{self.wrist_window[0]:.0f} to {self.wrist_window[1]:.0f} deg")

    def check_ft(self, tare=None):
        """Abort on a stale HEX-E stream, a protective stop, a load past the limits (relative to tare), or joint 5
        outside the wrist window."""
        self.check_wrist()
        t, v = self.be.ft.ring.latest()
        if t is None or self.now() - t > 0.1:
            self.be.robot.stopL(2.0)
            raise Abort("no HEX-E sample for 100 ms")
        if self.be.robot.stopped():
            raise Abort("protective or emergency stop")
        if tare is not None:
            d = v - tare
            if np.linalg.norm(d[:3]) > self.safety["abort_N"] or np.linalg.norm(d[3:]) > self.safety["abort_Nm"]:
                self.be.robot.stopL(3.0)
                raise Abort(f"load {np.linalg.norm(d[:3]):.1f} N, {np.linalg.norm(d[3:]):.2f} N m past the limits")

    def mean_ft(self, dur):
        t0 = self.now()
        while self.now() - t0 < dur:
            self.check_ft()
            self.sleep(0.01)
        t, v = self.ft_since(t0)
        return v.mean(axis=0), v.std(axis=0), len(t)

    def push_force(self, v, tare_v, d_B, R_BE):
        """Force (N) with which the indenter pushes the pad against the approach d_B (base), from the HEX-E's change."""
        f_E = np.linalg.solve(self.reg.K, v[:3] - tare_v[:3])
        return float(-(R_BE @ f_E) @ d_B)

    def in_box(self, T):
        if self.reg.T_BI is None:
            return True
        p = apply(inv(self.reg.T_BI), T[:3, 3])
        return np.all(np.abs(p[:2]) < self.safety["box_mm"]) and -20.0 < p[2] < self.safety["box_mm"]

    def goto(self, T, v=0.02, a=0.2, joint=False):
        if not self.in_box(T):
            raise Abort("target outside the box around the indenter")
        _, w0 = self.be.ft.ring.latest()
        if joint:                       # joint moves go to the IK branch of the homing pose, as the plan checked them
            q_ref = self.reg.q_home if getattr(self.reg, "q_home", None) is not None else self.be.robot.q()
            q = self.be.robot.ik(T, q_ref)
            if q is None:
                raise Abort("no IK solution")
            self.check_wrist(q, target=True)
            self.be.robot.moveJ(q, 0.4, 0.6, asynchronous=True)
        else:
            self.be.robot.moveL(T, v, a, asynchronous=True)
        t0 = self.now()
        self.sleep(0.05)
        lim = float(self.safety.get("travel_abort_N", 8.0))      # the payload's weight turns by at most 3.6 N
        while not self.be.robot.steady() or (self.now() - t0 < 0.2 and pose_error(self.be.robot.pose(), T)[0] > 1.0):
            self.check_ft()
            _, w = self.be.ft.ring.latest()
            if w0 is not None and np.linalg.norm(w[:3] - w0[:3]) > lim:
                self.be.robot.stopL(3.0)
                raise Abort(f"load of {np.linalg.norm(w[:3] - w0[:3]):.1f} N while travelling")
            self.sleep(0.01)
            if self.now() - t0 > 60.0:
                raise Abort("travel took over 60 s")
        e = pose_error(self.be.robot.pose(), T)
        if e[0] > 1.0 or e[1] > 0.5:
            raise Abort(f"the robot stopped {e[0]:.2f} mm, {e[1]:.2f} deg from its target")

    def T_BE_for(self, T_IP):
        return self.reg.T_BI @ T_IP @ inv(self.reg.T_EP())

    def T_BE_for_mount(self, T_IM):
        return self.reg.T_BI @ T_IM @ inv(self.reg.T_EM)

    def approach_dir(self):
        return -(self.reg.T_BI[:3, :3] @ EZ)

    def lifted(self, T, h_mm):
        """T moved up (along z_I) until the pad's lowest free-surface point stands h_mm above the indenter's top (at
        least 10 mm up)."""
        I = self.reg.T_BI
        pts = apply(inv(I) @ T @ self.reg.T_EP(), self.geo.pts)
        top = (self.geo.c_I + self.geo.R * EZ)[2]
        dz = max(10.0, h_mm + top - float(pts[:, 2].min()))
        return Tm(T[:3, :3], T[:3, 3] + I[:3, :3] @ EZ * dz)

    def travel(self, T_target):
        """From wherever the tool is to T_target near the indenter, as the plan checked it: up 10 mm, across, down; or,
        when the tool turns more than travel.turn_deg or moves more than travel.far_mm, up until the pad is travel.lift_mm
        above the indenter's top, across (in joint space when the turn exceeds travel.joint_deg), down."""
        cfg = dict(turn_deg=20.0, far_mm=60.0, lift_mm=60.0, joint_deg=45.0)
        cfg.update(self.plan.get("travel", {}))
        Tc = self.be.robot.pose()
        mm, deg = pose_error(Tc, T_target)
        if mm < 1e-3 and deg < 1e-3:
            return
        if deg > cfg["turn_deg"] or mm > cfg["far_mm"]:
            self.goto(self.lifted(Tc, cfg["lift_mm"]), 0.03)
            self.goto(self.lifted(T_target, cfg["lift_mm"]), 0.03, joint=deg > cfg["joint_deg"])
            self.goto(self.lifted(T_target, 0.0), 0.02)
        else:
            up = self.reg.T_BI[:3, :3] @ EZ
            self.goto(Tm(Tc[:3, :3], Tc[:3, 3] + up * 10.0), 0.02)
            self.goto(Tm(T_target[:3, :3], T_target[:3, 3] + up * 10.0), 0.02)
        self.goto(T_target, 0.01)

    def target_pose(self, a, standoff):
        """(flange pose at the action's rest, approach direction) for a plan action."""
        tg = a["target"]
        st = standoff + self.dry
        if tg["frame"] == "M":
            R = rot(EZ, a.get("twist", 0.0)) @ self.geo.R_ref @ rot_from_to(tg["n"], [1.0, 0.0, 0.0])
            c = self.geo.contact_point_I(st, tg.get("y_rod", 0.0))
            T_IM = Tm(R, c - R @ np.asarray(tg["p"], float))
            return self.T_BE_for_mount(T_IM), self.approach_dir()
        T_IP = self.geo.T_IP(tg["p"], tg["n"], a.get("twist", 0.0), st, a.get("offset", (0.0, 0.0)), tg.get("y_rod", 0.0))
        return self.T_BE_for(T_IP), self.approach_dir()

    # ---------------------------------------------------------------- the press primitive
    def press(self, a, T_rest, d_B, levels=(), stop_N=0.5, max_mm=3.0, v=0.0005, hold_s=0.6, rest_s=0.5, keep=False):
        """From T_rest along d_B at v (m/s) until the push passes each level (hold there), or stop_N, or max_mm; then
        back to T_rest (unless keep, which leaves the pad pressed). Returns the record of times."""
        rec = dict(id=a["id"], levels=[], t_rest0=None)
        tare_v, _, _ = self.mean_ft(rest_s)
        rec["t_rest0"] = [self.now() - rest_s, self.now()]
        rec["tare"] = tare_v
        R_BE = T_rest[:3, :3]
        T_end = Tm(R_BE, T_rest[:3, 3] + d_B * (max_mm))
        lv = sorted(set(float(x) for x in levels))
        if not lv or stop_N > lv[-1] + 1e-9:
            lv.append(float(stop_N))
        k = 0
        t_start = self.now()
        rec["t_move0"] = t_start
        self.be.robot.moveL(T_end, v, 0.05, asynchronous=True)
        f_hist = deque(maxlen=8)
        while True:
            self.sleep(0.004)
            self.check_ft(tare_v)
            t, val = self.be.ft.ring.latest()
            f_hist.append(self.push_force(val, tare_v, d_B, R_BE))
            f = float(np.mean(f_hist))
            if k < len(lv) and f >= lv[k]:
                self.be.robot.stopL(0.05)
                self.wait_steady()
                if k == len(lv) - 1:
                    rec["t_stop"] = self.now()
                    if hold_s and any(abs(lv[k] - float(x)) < 1e-9 for x in levels):
                        t0 = self.now()
                        self.hold(hold_s, tare_v)
                        rec["levels"].append(dict(N=lv[k], t=[t0, self.now()]))
                    break
                t0 = self.now()
                self.hold(hold_s, tare_v)
                rec["levels"].append(dict(N=lv[k], t=[t0, self.now()]))
                k += 1
                self.be.robot.moveL(T_end, v, 0.05, asynchronous=True)
                continue
            if self.be.robot.steady() and self.now() - t_start > 0.3:
                rec["t_stop"] = self.now()
                rec["reached_max"] = True
                break
        rec["T_stop"] = self.be.robot.pose()
        if keep:
            rec["tare_v"] = tare_v
            return rec
        self.be.robot.moveL(T_rest, 0.002, 0.05)
        t0 = self.now()
        self.mean_ft(rest_s)
        rec["t_rest1"] = [t0, self.now()]
        return rec

    def hold(self, dur, tare_v):
        t0 = self.now()
        while self.now() - t0 < dur:
            self.check_ft(tare_v)
            self.sleep(0.01)

    def turn_path(self, T0, centre_B, axis_B, degs, step_deg=0.5):
        """Flange poses turning T0 about (centre, axis) through the angles degs (deg), step_deg apart."""
        out, cur = [], 0.0
        for d in degs:
            n = max(1, int(math.ceil(abs(d - cur) / step_deg)))
            for j in range(1, n + 1):
                out.append(turn_about(T0, centre_B, axis_B, cur + (d - cur) * j / n))
            cur = d
        return out

    def run_turns(self, a, rec, centre_B, axis_B, degs, hold_s=0.4, step_deg=0.5):
        """In contact (after press(..., keep=True)): turn the pressed pose about (centre, axis) to each angle in degs
        (deg from the pressed pose), in steps of step_deg along a blended path, and hold at each."""
        T0 = rec["T_stop"]
        rec["turns"] = []
        cur = 0.0
        for d in degs:
            n = max(1, int(math.ceil(abs(d - cur) / step_deg)))
            path = [turn_about(T0, centre_B, axis_B, cur + (d - cur) * j / n) for j in range(1, n + 1)]
            self.be.robot.moveL_path(path, 0.004, 0.05, 0.0002, asynchronous=True)
            t_m = self.now()
            self.sleep(0.05)
            while not self.be.robot.steady() or (self.now() - t_m < 0.2 and pose_error(self.be.robot.pose(), path[-1])[0] > 0.05):
                self.check_ft(rec["tare_v"])
                self.sleep(0.005)
            cur = d
            self.check_ft(rec["tare_v"])
            t0 = self.now()
            self.hold(hold_s, rec["tare_v"])
            rec["turns"].append(dict(deg=d, t=[t0, self.now()], T_BE=self.be.robot.pose()))

    # ---------------------------------------------------------------- stages
    def actions(self, stage):
        return [a for a in self.plan["actions"] if a["stage"] == stage]

    def act(self, a, fn):
        if a["id"] in self.done:
            return None
        if self.args.step:
            self.be.operator(f"next: {a['id']} ({a['kind']})", kind="step")
        self.log.event(event="start", id=a["id"], stage=a["stage"], kind=a["kind"])
        gaps = getattr(self.be.ft, "gaps", [])
        n_gap = len(gaps)
        try:
            rec = fn(a)
            if len(gaps) > n_gap:                                # e.g. in a blocking joint move: the zero may have moved
                for g in gaps[n_gap:]:
                    self.log.event(event="ft_gap", id=a["id"], t0=g[0], t1=g[1], seq0=g[2], seq1=g[3], dFz_N=g[4])
                g = max(gaps[n_gap:], key=lambda g: g[1] - g[0])
                raise Abort(f"the HEX-E stream stopped for {g[1] - g[0]:.1f} s during the action (Fz then changed by "
                            f"{g[4]:+.1f} N)")
        except Abort as e:
            self.log.event(event="end", id=a["id"], status="abort", why=str(e))
            print(f"  {a['id']}: ABORT {e}", flush=True)
            self.n_abort += 1
            if self.n_abort >= 3:
                raise SystemExit("three aborts in a row: stopping the session (--resume continues it)")
            self.recover()
            return None
        self.n_abort = 0
        self.log.event(event="end", id=a["id"], status="ok", rec=rec)
        self.done.add(a["id"])
        return rec

    def recover(self):
        """After an abort: wait for the operator if the robot is stopped, then lift 5 mm away from the indenter."""
        if self.be.robot.stopped():
            self.be.operator("the robot is stopped: clear the protective stop on the pendant", kind="clear")
        try:
            Tc = self.be.robot.pose()
            self.be.robot.moveL(Tm(Tc[:3, :3], Tc[:3, 3] - self.approach_dir() * 5.0), 0.005, 0.05)
        except Exception as e:                                   # noqa: BLE001
            print("  recovery move failed:", e)

    def stage_preflight(self):
        self.log.event(event="start", id="preflight", stage="preflight", kind="check")
        self.sleep(2.0)
        t1 = self.now()
        tf, vf = self.ft_since(t1 - 2.0)
        tb, vb = self.board_since(t1 - 2.0)
        tr, vr = self.robot_since(t1 - 2.0)
        out = dict(ft_hz=len(tf) / 2.0, board_hz=len(tb) / 2.0, robot_hz=len(tr) / 2.0,
                   ft_std=vf.std(axis=0) if len(vf) else None,
                   board_std_uT=vb.reshape(-1, 5, 4)[:, :, 1:].std(axis=0) if len(vb) else None)
        print(f"preflight: HEX-E {out['ft_hz']:.0f} Hz (asked {self.args.ft_rate} Hz, filter {self.args.ft_filter}), "
              f"board {out['board_hz']:.0f} records/s, robot {out['robot_hz']:.0f} Hz")
        if len(vf):
            print("  HEX-E noise (1 sigma) N, N m:", np.round(vf.std(axis=0), 4))
        ft = self.be.ft
        if hasattr(ft, "n_late"):
            out.update(ft_timeouts=ft.n_timeout, ft_late=ft.n_late)
            print(f"  HEX-E reads: {ft.n_timeout} timed out, {ft.n_late} late datagrams dropped (0 and 0 on a clean link)")
        pl = self.plan.get("payload")
        if pl:
            self.be.robot.set_payload(pl["kg"], pl["cog_mm"])
            print(f"  payload set: {pl['kg']:.3f} kg at {pl['cog_mm']} mm from the flange")
        if out["ft_hz"] < 0.8 * self.args.ft_rate or out["board_hz"] < 50:      # a 100 Hz poll counts 99.5-100.5
            raise SystemExit("a stream is missing or slow: check the HEX-E's DAQ (IP, power) and the board's USB")
        self.log.event(event="end", id="preflight", status="ok", rec=out)

    def home_from_file(self):
        """The flange pose of a recorded seating on the indenter's key (--home-from FILE:LABEL, a JSON line with label and
        T_tcp_mm, the flange pose in mm as 4 x 4: the hand seatings of 2026-10-22), for runs with nobody at the robot."""
        path, _, label = self.args.home_from.rpartition(":")
        recs = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
        hit = [r for r in recs if r.get("label") == label]
        if not hit:
            raise SystemExit(f"--home-from: no record labelled {label!r} in {path}")
        return np.array(hit[-1]["T_tcp_mm"], float)

    def stage_homing(self):
        if "homing" in self.done and self.reg.T_BI is not None:
            return
        n = int(self.plan.get("homing", {}).get("repeats", 3))
        poses = []
        self.log.event(event="start", id="homing", stage="homing", kind="homing")
        if getattr(self.args, "home_from", None):
            poses = [self.home_from_file()]
            n = 0
            print(f"homing: from the recorded seating {self.args.home_from} (no freedrive, nothing moves)", flush=True)
        for k in range(n):
            self.be.robot.teach(True)
            self.be.operator(f"homing {k + 1}/{n}: in freedrive, seat the mount's socket on the indenter's key "
                             "(palmar face down, the finger along the indenter's +x), hold it seated", kind="seat")
            t0 = self.now()
            self.sleep(1.0)
            tr, vr = self.robot_since(t0)
            Ts = [ur_to_T(v[7:13]) for v in vr] if len(vr) else [self.be.robot.pose()]
            poses.append(mean_pose(Ts))
            self.be.robot.teach(False)
            up = poses[-1][:3, :3] @ (self.reg.T_EM[:3, :3] @ np.array([-1.0, 0.0, 0.0]))   # out of the key: -x_M
            Tc = self.be.robot.pose()
            self.be.robot.moveL(Tm(Tc[:3, :3], Tc[:3, 3] + up * 12.0), 0.01, 0.1)
        Th = mean_pose(poses)
        spread = [pose_error(Th, T) for T in poses]
        self.reg.T_BE_home = Th
        self.reg.q_home = self.be.robot.ik(Th, self.be.robot.q())
        self.reg.T_BI = Th @ self.reg.T_EM @ inv(self.geo.T_IM_home)
        self.reg.notes["homing_spread_mm_deg"] = spread
        print(f"homing: {len(poses)} seatings within {max(s[0] for s in spread):.3f} mm, {max(s[1] for s in spread):.3f} deg "
              "of their mean")
        self.check_placement()
        self.save_reg()
        self.log.event(event="end", id="homing", status="ok", rec=dict(T_BE_home=Th, poses=poses, spread=spread))
        self.done.add("homing")

    def check_placement(self, max_mm=20.0, max_yaw=5.0, max_tilt=2.0):
        """The plan's reach and clearance checks hold for the placement it was built on (checks.placement_mm_deg: x, y, z
        in the base frame, yaw of x_I about z_B). Homing measures the real one; past the limits the operator moves the
        plywood, rebuilds the plan, or types 'continue'."""
        pl = self.plan.get("checks", {}).get("placement_mm_deg")
        if pl is None:
            return
        p, R = self.reg.T_BI[:3, 3], self.reg.T_BI[:3, :3]
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0))))
        dp = p - np.array(pl[:3], float)
        dyaw = (yaw - pl[3] + 180.0) % 360.0 - 180.0
        self.reg.notes["placement_offset_mm_deg"] = dict(d_mm=dp.tolist(), d_yaw=dyaw, tilt=tilt)
        print(f"  indenter at ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm, yaw {yaw:.1f} deg, tilt {tilt:.2f} deg; the plan has "
              f"({pl[0]:.0f}, {pl[1]:.0f}, {pl[2]:.0f}) mm, yaw {pl[3]:.0f} deg")
        if np.linalg.norm(dp) <= max_mm and abs(dyaw) <= max_yaw and tilt <= max_tilt:
            return
        if getattr(self.args, "home_from", None):                # nobody at the robot to answer a prompt
            raise SystemExit(f"stopped at the placement check: the indenter is {np.linalg.norm(dp):.0f} mm, {dyaw:+.1f} "
                             f"deg of yaw and {tilt:.1f} deg of tilt from the plan's placement")
        ans = self.be.operator(
            f"the indenter is {np.linalg.norm(dp):.0f} mm, {dyaw:+.1f} deg of yaw and {tilt:.1f} deg of tilt from the plan's "
            f"placement (limits {max_mm:.0f} mm, {max_yaw:.0f} deg, {max_tilt:.0f} deg), where its reach and clearance were not "
            f"checked. Move the plywood by ({-dp[0]:+.0f}, {-dp[1]:+.0f}) mm in the base frame and run again, or rebuild "
            f"the plan with --placement {p[0]:.0f},{p[1]:.0f},{p[2]:.0f},{yaw:.0f}. Type continue to go on", kind="placement")
        if str(ans).strip().lower() != "continue":
            raise SystemExit("stopped at the placement check; homing runs again on the next start")

    def stage_gravity(self, stage="gravity"):
        acts = self.actions(stage)
        if not acts:
            return
        fresh = any(a["id"] not in self.done for a in acts)     # one sweep, one zero: a resume re-tares the HEX-E
        if fresh:
            self.done -= {a["id"] for a in acts}
        R, F, Tq, B, Tt = [], [], [], [], []
        for a in acts:
            def one(a):
                T_IP = Tm(rot(a["axis"], a["deg"]) @ self.geo.R_IP([1.0, 0.0, 0.0], 0.0), None)
                p = np.array(a["station_I"], float)
                T_IP[:3, 3] = p - T_IP[:3, :3] @ np.array(a["p"], float)
                T = self.T_BE_for(T_IP)
                q = self.be.robot.ik(T, self.reg.q_home if self.reg.q_home is not None else self.be.robot.q())
                if q is None:
                    raise Abort("orientation out of reach")
                self.check_wrist(q, target=True)
                self.be.robot.moveJ(q, 0.4, 0.6)
                self.sleep(0.8)
                t0 = self.now()
                m, s, n = self.mean_ft(1.5)
                tb, vb = self.board_since(t0)
                return dict(R_BE=self.be.robot.pose()[:3, :3], ft=m, ft_std=s, n=n, t=[t0, self.now()],
                            board=vb.mean(axis=0) if len(vb) else None)
            rec = self.act(a, one)
            if rec is None and not fresh:                        # the whole sweep ended ok in an earlier run
                for e in self.log.records("end"):
                    if e.get("id") == a["id"] and e.get("status") == "ok":
                        rec = e["rec"]
            if rec is not None:
                R.append(np.array(rec["R_BE"]))
                Tt.append(float(np.mean(rec["t"])))
                F.append(np.array(rec["ft"])[:3])
                Tq.append(np.array(rec["ft"])[3:])
                B.append(rec.get("board"))
        if len(R) < 6:
            print(f"{stage}: only {len(R)} orientations, no fit")
            return
        fit = fit_gravity(np.array(R), np.array(F), np.array(Tq), t=np.array(Tt))
        print(f"{stage}: weight {fit['weight_N']:.3f} N ({fit['mass_kg'] * 1e3:.0f} g), centre of mass {np.round(fit['com_mm'], 1)} mm "
              f"(from the HEX-E's origin), sign {fit['sign']:+.0f}, HEX-E axes turned {fit['z_turn_deg']:+.2f} deg about the "
              f"tool axis ({fit['turn_deg']:.2f} in all), residual {fit['rms_f_N']:.4f} N, {fit['rms_t_Nm']:.5f} N m; "
              f"drift during the {max(Tt) - min(Tt):.0f} s sweep {np.round(fit['v_f'], 3)} N/min, "
              f"{np.round(fit['v_t'], 4)} N m/min")
        lim = float(getattr(self.args, "gravity_max_rms", 0.1))
        want = getattr(self.args, "gravity_expect_N", None)
        if stage == "gravity" and want is not None and abs(fit["weight_N"] / float(want) - 1.0) > 0.05:
            self.reg.notes["gravity_failed"] = {k: fit[k] for k in ("weight_N", "rms_f_N", "rms_t_Nm")}
            self.save_reg()
            raise SystemExit(f"gravity: weight {fit['weight_N']:.3f} N, over 5 % from the expected {float(want):.3f} N: "
                             "the HEX-E's gain or the payload changed; stopping before any contact")
        if fit["rms_f_N"] > lim:
            self.reg.notes[f"{stage}_failed"] = {k: fit[k] for k in ("weight_N", "rms_f_N", "rms_t_Nm")}
            self.save_reg()
            msg = (f"{stage}: residual {fit['rms_f_N']:.3f} N over {lim:.3f} N (Table 2: under 0.02 N): a cable pulls "
                   "on the tool, or the HEX-E's zero moved during the sweep")
            if stage == "gravity":
                raise SystemExit(msg + "; stopping before any contact")
            print("WARNING " + msg, flush=True)
        if stage == "gravity":
            self.reg.grav = {k: fit[k] for k in ("A", "of", "Bt", "ot", "v_f", "v_t", "t_mean")}
            self.reg.K, self.reg.Kt = fit["K"], fit["Kt"]
            self.reg.T_ES = Tm(fit["R_SE"].T, self.reg.T_ES[:3, 3])
            self.reg.notes["gravity"] = {k: fit[k] for k in ("weight_N", "mass_kg", "com_mm", "sign", "turn_deg", "z_turn_deg",
                                                              "rms_f_N", "rms_t_Nm", "skew_of_K", "sv", "n", "v_f", "v_t")}
        else:
            self.reg.notes[stage] = {k: fit[k] for k in ("weight_N", "com_mm", "z_turn_deg", "rms_f_N", "rms_t_Nm", "v_f", "v_t")}
        self.save_reg()

    def touch(self, a, kind):
        st = float(a.get("standoff", 1.5))
        T_rest, d = self.target_pose(a, st)
        self.travel(T_rest)
        rec = self.press(a, T_rest, d, levels=(), stop_N=float(a.get("stop_N", 0.5)),
                         max_mm=st + float(a.get("beyond_mm", 1.2)), v=float(a.get("v", 0.00015)), hold_s=0.0, rest_s=0.5)
        if self.dry:                                             # no contact by design: the approach is the record
            return rec
        t0, t1 = rec["t_move0"], rec["t_stop"]
        tf, vf = self.ft_since(t0, t1)
        tr, vr = self.robot_since(t0 - 0.05, t1 + 0.05)
        if len(tf) < 20 or len(tr) < 5:
            raise Abort("too few samples in the approach")
        P = np.array([ur_to_T(v[7:13])[:3, 3] for v in vr])
        s_r = (P - T_rest[:3, 3]) @ d
        s = np.interp(tf, tr, s_r)
        y = np.array([self.push_force(v, rec["tare"], d, T_rest[:3, :3]) for v in vf])
        fo = fit_onset(s, y, y_max=float(a.get("fit_N", 0.4)),
                       p_set=(1.0,) if kind == "mount" else (1.0, 1.25, 1.5, 1.75, 2.0))
        if fo is None:
            raise Abort("no contact onset in the approach")
        s0, p, rms, nc = fo
        T_on = Tm(T_rest[:3, :3], T_rest[:3, 3] + d * s0)
        tch = dict(id=a["id"], kind=kind, T_BE=T_on, s0=s0, p=p, rms_N=rms, n_contact=nc, target=a["target"],
                   expected_mm=st + 0.0)
        self.touches.append(tch)
        self.save_reg()
        print(f"  {a['id']}: onset {s0 - st:+.3f} mm from the expected (power {p:.2f}, fit {rms * 1e3:.1f} mN)", flush=True)
        return dict(rec, onset=tch)

    def stage_touch(self, stage, kind):
        for a in self.actions(stage):
            self.act(a, lambda a: self.touch(a, kind))

    def stage_register(self, use_pivots=False):
        if self.dry:
            print("register: skipped in a dry run")
            return
        if self.reg.T_BI is None:
            print("register: no homing yet")
            return
        if not self.touches:
            print("register: no touches")
            return
        piv = self.pivots if use_pivots else ()
        fit = fit_registration(self.geo, self.reg, self.touches, piv)
        self.reg.T_EM, self.reg.T_MP = fit["T_EM"], fit["T_MP"]
        c = fit["c_B"]
        self.reg.T_BI = Tm(self.reg.T_BI[:3, :3], c - self.reg.T_BI[:3, :3] @ self.geo.c_I)
        self.reg.notes["register" + ("_pivot" if use_pivots else "")] = dict(
            {k: fit[k] for k in ("rms_pad_mm", "rms_face_mm", "n_pad", "n_face", "n_pivot", "tcp_change_mm_deg",
                                 "centre_change_mm", "sd", "x")}, T_EM=fit["T_EM"], T_MP=fit["T_MP"], c_B=fit["c_B"])
        print(f"register{' with pivots' if use_pivots else ''}: pad touches {fit['rms_pad_mm']:.3f} mm rms ({fit['n_pad']}), face "
              f"{fit['rms_face_mm']:.3f} mm ({fit['n_face']}); TCP moved {fit['tcp_change_mm_deg'][0]:.3f} mm, "
              f"{fit['tcp_change_mm_deg'][1]:.3f} deg; sphere centre moved {fit['centre_change_mm']:.3f} mm")
        Tp = self.reg.T_EP()
        print("  TCP of the pad frame on the flange (for the pendant): " +
              ", ".join(f"{v:.5f}" for v in T_to_ur(Tp)) + "  (m, rotation vector)")
        self.save_reg()
        self.log.event(event="fit", id="register" + ("_pivot" if use_pivots else ""), rec=fit)

    def board_level(self, rec, N):
        """Board change (15, uT, chip x y z) at exactly N newtons of measured push: from the holds' means (each against
        the rests before and after the press, interpolated to the hold's time), linear between the two holds nearest N;
        with the mean tangential force there (N, flange frame)."""
        if not rec.get("t_rest0") or not rec.get("t_rest1") or len(rec["levels"]) < 1:
            return None
        r0 = self.board_since(*rec["t_rest0"])[1].reshape(-1, 5, 4)[:, :, 1:].mean(axis=0).ravel()
        r1 = self.board_since(*rec["t_rest1"])[1].reshape(-1, 5, 4)[:, :, 1:].mean(axis=0).ravel()
        ta, tb_ = np.mean(rec["t_rest0"]), np.mean(rec["t_rest1"])
        pts = []
        d = self.approach_dir()
        R_BE = np.array(rec["T_stop"])[:3, :3] if rec.get("T_stop") is not None else self.be.robot.pose()[:3, :3]
        for lv in rec["levels"]:
            tb, vb = self.board_since(*lv["t"])
            tf, vf = self.ft_since(*lv["t"])
            if not len(vb) or not len(vf):
                continue
            tm = np.mean(lv["t"])
            rest = r0 + (r1 - r0) * (tm - ta) / max(tb_ - ta, 1e-6)
            f_E = np.linalg.solve(self.reg.K, vf[:, :3].mean(axis=0) - np.array(rec["tare"])[:3])
            push = float(-(R_BE @ f_E) @ d)
            pts.append((push, vb.reshape(-1, 5, 4)[:, :, 1:].mean(axis=0).ravel() - rest, f_E))
        if not pts:
            return None
        pts.sort(key=lambda x: x[0])
        if len(pts) == 1:
            return pts[0][1] * N / pts[0][0], pts[0][2]
        (f1, b1, e1), (f2, b2, e2) = pts[-2], pts[-1]
        w = (N - f1) / (f2 - f1)
        return b1 + (b2 - b1) * w, e1 + (e2 - e1) * w

    def stage_pivot(self):
        acts = self.actions("pivot")
        if not acts:
            return
        recs = {}
        for a in acts:
            def one(a):
                T_rest, d = self.target_pose(a, 1.5)
                self.travel(T_rest)
                r = self.press(a, T_rest, d, levels=a.get("levels", (1.5, 3.0)), stop_N=a.get("stop_N", 3.0),
                               max_mm=4.0, v=0.0004, hold_s=float(a.get("hold_s", 1.5)))
                r["T_rest"] = T_rest
                return r
            r = self.act(a, one)
            if r is not None:
                recs[a["id"]] = (a, r)
        if self.dry:
            return
        lvl = float(self.plan.get("pivot", {}).get("level_N", 1.5))
        ref = [v for v in recs.values() if v[0].get("role") == "ref"]
        offs = [v for v in recs.values() if v[0].get("role") == "offset"]
        turns = [v for v in recs.values() if v[0].get("role") == "turn"]
        if not ref or len(offs) < 4:
            print("pivot: missing presses")
            return
        b0, f0 = self.board_level(ref[0][1], lvl)
        cols, rows = [], []
        for a, r in offs:
            bl = self.board_level(r, lvl)
            if bl is not None:
                rows.append(np.array(a["offset_P"]))
                cols.append(bl[0] - b0)
        J, *_ = np.linalg.lstsq(np.array(rows), np.array(cols), rcond=None)       # (2, 15): change per mm along t1, t2
        J = J.T
        out = []
        t1, t2 = (np.array(self.plan["pivot"][k], float) for k in ("t1", "t2"))
        for a, r in turns:
            bl = self.board_level(r, lvl)
            if bl is None:
                continue
            b, f_E = bl
            dp = np.linalg.lstsq(J, b - b0, rcond=None)[0]
            ft = float(np.linalg.norm(f_E - (f_E @ f0) / (f0 @ f0) * f0)) if np.any(f0) else float("nan")
            self.pivots.append(dict(id=a["id"], T_BE=r["T_rest"], T_BE_ref=ref[0][1]["T_rest"], t1=t1, t2=t2, dp_board=dp,
                                    resid_uT=float(np.linalg.norm(J @ dp - (b - b0))), shear_N=ft))
            out.append((a["twist"], dp, ft))
            print(f"  pivot {a['id']} twist {a['twist']:+.0f} deg: the board places it {dp[0]:+.3f}, {dp[1]:+.3f} mm from the "
                  f"reference (shear {ft:.3f} N, fit residual {self.pivots[-1]['resid_uT']:.1f} uT)")
        # the registration's own prediction of the same offsets: the pivot checks it (folding the board's offsets into
        # the fit made the contact point worse in simulation: the board's offsets carry about 0.1 mm of bias at k 0.35,
        # and the robot's error changes with its joints, which one tool pose cannot follow)
        for pv in self.pivots:
            T_EP = self.reg.T_EP()
            c = apply(self.reg.T_BI, self.geo.c_I)
            q0 = apply(inv(np.array(pv["T_BE_ref"]) @ T_EP), c)
            q1 = apply(inv(np.array(pv["T_BE"]) @ T_EP), c)
            pv["dp_registration"] = [float((q1 - q0) @ t1), float((q1 - q0) @ t2)]
            dd = np.array(pv["dp_board"]) - np.array(pv["dp_registration"])
            print(f"  {pv['id']}: board {pv['dp_board'][0]:+.3f} {pv['dp_board'][1]:+.3f} mm, registration "
                  f"{pv['dp_registration'][0]:+.3f} {pv['dp_registration'][1]:+.3f} mm, apart {np.linalg.norm(dd):.3f} mm")
        worst = max((np.linalg.norm(np.array(pv["dp_board"]) - np.array(pv["dp_registration"])) for pv in self.pivots), default=0.0)
        self.reg.notes["pivot_check"] = dict(worst_mm=float(worst), pivots=self.pivots)
        if worst > 0.5:
            print(f"  WARNING: the board and the registration disagree by {worst:.2f} mm: check the TCP (Section 3 of the page)")
        self.log.save("pivots.json", self.pivots)
        self.log.event(event="fit", id="pivot", rec=dict(J=J, turns=out, worst_mm=worst))
        self.save_reg()
        if getattr(self.args, "pivot_fit", False):
            self.stage_register(use_pivots=True)

    def grid_press(self, a):
        T_rest, d = self.target_pose(a, float(a.get("standoff", 1.5)))
        self.travel(T_rest)
        r = self.press(a, T_rest, d, levels=a.get("levels", (0.25, 0.5, 1.0, 2.0)), stop_N=a.get("stop_N", max(a.get("levels", [2.0]))),
                       max_mm=float(a.get("standoff", 1.5)) + float(a.get("max_depth_mm", 2.5)), v=float(a.get("v", 0.0005)),
                       hold_s=float(a.get("hold_s", 0.6)))
        r["T_rest"] = T_rest
        return r

    def stage_presses(self, stage):
        acts = self.actions(stage)
        for k, a in enumerate(acts):
            if a["kind"] == "press":
                r = self.act(a, self.grid_press)
            elif a["kind"] in ("slide", "roll", "twist"):
                r = self.act(a, self.contact_turns)
            else:
                continue
            if r is not None and r.get("levels"):
                top = r["levels"][-1]["N"]
                print(f"  {a['id']} ({k + 1}/{len(acts)}): to {top:.2f} N", flush=True)

    def contact_turns(self, a):
        """Press to a['press_N'] and hold, then the moves in turn, each back to the pressed pose at its end: slide =
        turns about the sphere's centre (the contact slides along dir_deg in I at constant indentation, R deg / 57.3 mm
        per degree), roll = the same axis through the contact point (the contact rolls), twist = about the vertical
        through the contact (torsion)."""
        st = float(a.get("standoff", 1.5))
        T_rest, d = self.target_pose(a, st)
        self.travel(T_rest)
        r = self.press(a, T_rest, d, levels=(a["press_N"],), stop_N=a["press_N"], max_mm=st + 3.0, v=0.0004, hold_s=0.5, keep=True)
        I = self.reg.T_BI
        c_B = apply(I, self.geo.c_I)
        top_B = apply(I, self.geo.c_I + self.geo.R * EZ)
        r["moves"] = []
        try:
            for mv in a["moves"]:
                ang = math.radians(mv.get("dir_deg", 0.0))
                u_I = np.array([math.cos(ang), math.sin(ang), 0.0])
                if a["kind"] == "slide":
                    centre, axis = c_B, I[:3, :3] @ np.cross(EZ, u_I)
                elif a["kind"] == "roll":
                    centre, axis = top_B, I[:3, :3] @ np.cross(EZ, u_I)
                else:
                    centre, axis = top_B, I[:3, :3] @ EZ
                self.run_turns(a, r, centre, axis, list(mv["degs"]) + [0.0], hold_s=float(a.get("hold_s", 0.4)))
                r["moves"].append(dict(dir_deg=mv.get("dir_deg", 0.0), centre_B=centre, axis_B=axis, turns=r.pop("turns")))
        finally:
            T_now = self.be.robot.pose()
            self.be.robot.moveL(Tm(T_now[:3, :3], T_now[:3, 3] - d * (st + 0.5)), 0.002, 0.05)
        t0 = self.now()
        self.mean_ft(0.5)
        r["t_rest1"] = [t0, self.now()]
        r["T_rest"] = T_rest
        return r

    # ---------------------------------------------------------------- the whole
    def run(self, stages):
        for s in stages:
            print(f"== {s}", flush=True)
            if s == "preflight":
                self.stage_preflight()
            elif s == "homing":
                self.stage_homing()
            elif s == "gravity":
                self.stage_gravity("gravity")
            elif s == "mount":
                self.stage_touch("mount", "mount")
            elif s == "pad":
                self.stage_touch("pad", "pad")
            elif s == "register":
                self.stage_register()
            elif s == "pivot":
                self.stage_pivot()
            elif s in ("grid", "shear", "roll", "twist", "rod"):
                self.stage_presses(s)
            elif s == "final":
                self.stage_presses("final")
                self.stage_gravity("final_gravity")
            else:
                raise SystemExit(f"unknown stage {s}")
        self.log.save("registration.json", self.reg.to_json())


STAGES = ("preflight", "homing", "gravity", "mount", "pad", "register", "pivot", "grid", "shear", "roll", "twist", "final")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--robot", default="192.168.0.4")
    ap.add_argument("--ft", default="192.168.0.5", help="the HEX-E DAQ (magpie_force_control uses 192.168.0.3)")
    ap.add_argument("--ft-rate", type=int, default=100, choices=(50, 100, 250, 500), help="HEX-E poll rate, Hz")
    ap.add_argument("--ft-filter", type=int, default=3, choices=range(7), help="the DAQ's low-pass: 0 none, 1 500 Hz, "
                    "2 150 Hz, 3 50 Hz (default: nothing above the 100 Hz poll's 50 Hz folds in), 4 15 Hz, 5 5 Hz, 6 1.5 Hz")
    ap.add_argument("--ft-stream", action="store_true", help="ask the DAQ to stream instead of polling it")
    ap.add_argument("--board", default=None, help="serial port of the board (default: found by id)")
    ap.add_argument("--home-from", default=None, help="FILE:LABEL, a recorded seating on the key (JSON lines with label "
                    "and T_tcp_mm) in place of the three freedrive seatings: for runs with nobody at the robot")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--wrist-window", default="30,100", help="deg: joint 5 stays within LO,HI in every move and target "
                    "(the lab plan of 2026-10-22 needs 37 to 90 deg outside the mount stage)")
    ap.add_argument("--gravity-expect-N", type=float, default=None, help="N: the weight of an earlier sweep with the same "
                    "tool; a fitted weight over 5 %% from it stops the session before any contact (2026-10-22: 1.567 N "
                    "in first2, before the HEX-E's cable port was squeezed)")
    ap.add_argument("--gravity-max-rms", type=float, default=0.1, help="N: the gravity sweep's force residual past which "
                    "the session stops before any contact (Table 2 expects under 0.02 N)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dry", type=float, default=0.0, help="mm added to every standoff: no contact")
    ap.add_argument("--step", action="store_true")
    ap.add_argument("--pivot-fit", action="store_true", help="fold the pivot's offsets into the registration (off: a check)")
    ap.add_argument("--logs", default=str(Path.cwd() / "logs"))
    ap.add_argument("--name", default="session")
    ap.add_argument("--sim", action="store_true", help="run against ur5_tip_calibration_sim.SimBackend")
    ap.add_argument("--sim-seed", type=int, default=0)
    ap.add_argument("--sim-truth", default=None, help="json of the simulator's errors (ur5_tip_calibration_sim.Truth)")
    a = ap.parse_args(argv)
    plan = json.loads(Path(a.plan).read_text())
    if plan.get("version") != PLAN_VERSION:
        raise SystemExit(f"plan version {plan.get('version')} (this program reads {PLAN_VERSION})")
    if a.sim:
        sys.path.insert(0, str(HERE))
        import ur5_tip_calibration_sim as SIM
        be = SIM.SimBackend(plan, seed=a.sim_seed, truth=a.sim_truth, ft_rate=a.ft_rate, ft_filter=a.ft_filter)
        log = SessionLog(a.logs, a.name, a.resume, clock=be.now)
        be.attach(log)
    else:
        log = SessionLog(a.logs, a.name, a.resume)
        be = RealBackend(a, log)
    log.save("plan.json", plan) if not a.resume else None
    log.save("session.json", dict(args=vars(a), started=time.strftime("%Y-%m-%d %H:%M:%S"), stages=a.stages,
                                  ft_columns="t_host seq status Fx Fy Fz Tx Ty Tz (N, N m)",
                                  robot_columns="t_host t_ctrl q0..q5 (rad) x y z (m) rx ry rz safety_mode",
                                  board_dtype="t <f8, r <f4 x 20 (5 chips x [T C, x, y, z uT])"))
    ses = Session(plan, be, log, a)
    try:
        ses.run([s for s in a.stages.split(",") if s])
    except KeyboardInterrupt:
        print("\ninterrupted: stopping the robot")
        try:
            be.robot.stopL(2.0)
        except Exception:                                        # noqa: BLE001
            pass
    finally:
        log.save("registration.json", ses.reg.to_json())
        ft = getattr(be, "ft", None)
        if hasattr(ft, "n_late"):
            gaps = [list(g) for g in getattr(ft, "gaps", [])]
            log.event(event="ft_reads", n=ft.n, timeouts=ft.n_timeout, late=ft.n_late, repeats=ft.n_repeat, gaps=gaps)
            print(f"HEX-E: {ft.n} samples, {ft.n_timeout} timed out, {ft.n_late} late datagrams dropped, "
                  f"{len(gaps)} gaps over 0.5 s" + "".join(f"; {g[1] - g[0]:.1f} s, Fz {g[4]:+.1f} N" for g in gaps))
        be.close()
        log.close()
        print(f"session in {log.dir}")


if __name__ == "__main__":
    main()
