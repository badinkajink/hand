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
registration. --dry 15 shifts every target 15 mm away from the indenter along its approach (no contact: reach, cables);
in probe mode back along the stem instead.
--step asks before each action. Ctrl-C stops the robot (stopL) and closes the files.

Probe mode (a plan with geometry.probe, from scripts/ur5_probe_plan.py; the inverted rig of 2026-10-30): a sphere on the
HEX-E presses a fingertip held still on a seat of the strip. Frame I is then the strip's frame A (homing seats the probe's
socket, the fork's, on a key of the strip), the pad stands at T_AP in A and the sphere's centre at c_M in M. A target gives
the pad point, its normal, the push direction and the mount's rotation in P; the sphere's centre goes to
T_BI T_AP (p + R n - standoff d). The travels are the plan's (travel.probe: straight, joint or by way of the hub, as the
workstation's replay checked them), the pad touch-offs fit T_AP and c_M (register), and the board is the seat's (its mux
channel geometry.seat.held_channel on --held-board). The wall stage touches the cassette's flat side walls at two heights
each, for the seat's turn about z_P (the tip's long axis), which the pad's touches barely see; it aims with the pad's fit,
so a probe session runs --stages preflight,homing,pad,register,wall,register,grid. The gravity, mount, pivot and hover
stages are the tip mode's.

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
import queue
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
FAST_STOP_N = 0.3                   # N of push that ends a fast approach early (the simulator's HEX-E noise reached 0.16 N over 8 samples)


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
    U_, _, Vt = np.linalg.svd(np.asarray(R, float))         # the nearest rotation first: near 180 deg the formula below
    R = U_ @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U_ @ Vt))]) @ Vt   # turns a plan's 7-decimal rounding (7e-6 off
    #                                                         orthonormal) into a degree (2026-11-02, the probe straight
    #                                                         down at 178.6 deg: every IK target 1.0 deg off its pose)
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
        self.T_IM_home = np.array(g["T_IM_home"], float)   # probe mode: the strip key's T_AM_home (I is the strip's A)
        self.probe = g.get("probe")
        if self.probe is not None:                          # probe mode: the sphere on the HEX-E, the pad on its seat
            self.kind = self.probe.get("kind", "sphere")
            self.c_M = np.array(self.probe["centre_M"], float)
            self.R = float(self.probe["radius"])
            if self.kind == "rod":                          # a rod (11-03): c_M a point of its axis under the tool's axis,
                self.axis_M = np.array(self.probe.get("axis_M", [0.0, 1.0, 0.0]), float)   # the axis along axis_M
                self.axis_M /= np.linalg.norm(self.axis_M)
                self.half_len = float(self.probe["half_len"])
            self.c_I, self.rod_axis = None, None
            self.seat = g["seat"]
            self.T_AP0 = np.array(self.seat["T_AP"], float)
        else:
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
        if self.probe is not None and self.kind == "rod":   # the surface densely sampled for the rod's line distance:
            self.dense = np.vstack([self.pts, (A + B + C) / 3.0, (A + B) / 2.0, (B + C) / 2.0, (C + A) / 2.0])   # vertices,
#                                                             centroids and edge midpoints, about 0.25 mm apart
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

    def line_distance(self, c, a):
        """Probe mode with a rod: the least distance (mm) of the pad's surface from the rod's axis, the line through c
        along a (P), over the surface within half_len of c along the line."""
        r = self.dense - np.asarray(c, float)
        y = r @ a
        m = np.abs(y) <= self.half_len
        if not m.any():
            return float("inf")
        rp = r[m] - y[m, None] * a[None]
        return float(np.sqrt(np.min(np.einsum("ij,ij->i", rp, rp))))

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
        self.T_AP = geo.T_AP0.copy() if geo.probe is not None else None   # probe mode: the held pad on its seat (A)
        self.c_M = geo.c_M.copy() if geo.probe is not None else None      # probe mode: the sphere's centre (M)

    def T_EP(self):
        return self.T_EM @ self.T_MP

    def T_BP(self):
        """Probe mode: the held pad in the base."""
        return self.T_BI @ self.T_AP

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


REG_SIGMA_PROBE = dict(pad_mm=0.03, wall_mm=0.02, wall_offset_mm=0.1, seat_rot_deg=(1.0, 1.0, 0.5),
                       seat_t_mm=(0.5, 0.5, 0.5), probe_mm=(0.3, 0.3, 0.3))
# the roll about z_P, the tip's long axis: four M2 on a 16 mm square clamp it, 0.05-0.1 mm of print across them rolls it
# 0.2-0.4 deg (workstation, 10-30 15:0x); the pad's touches barely see it, so the prior holds most of it. The side walls
# see it: two touches 6 mm apart on each flat wall of the cap, 0.02 mm each (the onset on PLA and the face's flatness).
# Each wall's place along its normal is free within wall_offset_mm (the print), so the walls' tilt enters the fit and
# their width, which would pull the sphere's centre along the stems, mostly does not
REG_DROP_MM = 0.5       # probe mode: register leaves out a pad touch standing this far off the pad under the fit, the
REG_DROP_MAX = 3        # worst first and at most 3, and stops the session before the grid when the rest stand over
REG_RMS_MAX_MM = 0.15   # REG_RMS_MAX_MM rms. On 11-03 s1sph2's pad-tip-349-o35-a270 met the tip's body 4.7 mm early: the fit
#                         with it stood 0.76 mm rms with the pad 5.7 mm off, its 19 others fit to 0.051 mm, and the grid
#                         met the pad early on most presses, two travels aborting at 8.4 and 8.8 N
EARLY_STOP = 3          # contacts before the slow approach that stop a session (none in the 2,400 actions of 11-02 and 11-03)
MISS_STOP = 3           # presses with no level reached that stop a session (none in those runs; s1sph2 had 8 in a row)
PUSH_AVG = dict(tip=8, probe=20)    # the push's running mean: the tip mode's over the last 8 passes of its 250 Hz loop (about
#                                     3 HEX-E samples), the probe mode's over 20 HEX-E samples (0.2 s), tested only when
#                                     full. A probe pushes along the HEX-E's z (0.21 N a sample at --ft-filter 3, five times x
#                                     and y), and the tip mode's mean crossed 0.3 N in free air (the simulator, 10-30)


def fit_probe_registration(geo: Geometry, reg: Registration, touches, sig=REG_SIGMA_PROBE):
    """Probe mode: the held pad on its seat (T_AP, in the strip's frame A, which homing put at T_BI) and the sphere's
    centre on the probe (c_M) from the touches (flange pose at the onset of contact), with the seat's and the probe's
    CAD as priors. A pad touch's residual is the tip mode's, the sphere's centre in P against the pad's surface, with
    q = (T_BI T_AP)^-1 T_BE T_EM c_M; a wall touch's is q's height over the wall's plane (the target's p and n, P) less R
    and less that wall's offset, a parameter per wall with the prior wall_offset_mm. Turns of the probe about the pad's
    normal leave the pad's residual alike, so c_M comes from touches whose stems lean differently. The priors stand at the
    plan's T_AP and c_M, so the fit runs again as touches are added (after the walls). Returns a dict with the new T_AP
    and c_M and the residuals."""
    pad = [t for t in touches if t["kind"] == "pad"]
    wall = [t for t in touches if t["kind"] == "wall"]
    T_AP0, c_M0 = geo.T_AP0, geo.c_M
    T_BA, T_EM = reg.T_BI, reg.T_EM
    Ts = [np.array(t["T_BE"], float) @ T_EM for t in pad]
    planes, Tw = {}, []
    for t in wall:                                           # one offset per plane (same normal and height in P)
        n = np.asarray(t["target"]["n"], float)
        n = n / np.linalg.norm(n)
        p = np.asarray(t["target"]["p"], float)
        k = planes.setdefault((tuple(np.round(n, 3)), round(float(n @ p), 2)), len(planes))
        Tw.append((np.array(t["T_BE"], float) @ T_EM, p, n, 9 + k))
    nx = 9 + len(planes)
    sd = np.r_[np.radians(sig["seat_rot_deg"]), sig["seat_t_mm"], sig["probe_mm"], [sig["wall_offset_mm"]] * len(planes)]

    def unpack(x):
        return perturb(T_AP0, x[0:6]), c_M0 + x[6:9]

    def resid(x):
        T_AP, c_M = unpack(x)
        T_PB = inv(T_BA @ T_AP)
        r = [(geo.surface_distance(apply(T_PB @ T, c_M))[0] - geo.R) / sig["pad_mm"] for T in Ts]
        r += [((apply(T_PB @ T, c_M) - p) @ n - geo.R - x[k]) / sig["wall_mm"] for T, p, n, k in Tw]
        return np.r_[r, x / sd]

    x, r, cov = lm(resid, np.zeros(nx))
    T_AP, c_M = unpack(x)
    n, m = len(pad), len(wall)
    wall_turn_deg = {}                                       # each wall alone: the turn about z_P its own touches give
    if m:
        dz = np.zeros(nx)
        dz[2] = 1e-6
        rw = r[n:n + m] * sig["wall_mm"]
        Jz = (resid(x + dz)[n:n + m] * sig["wall_mm"] - rw) / 1e-6
        for k in planes.values():
            ix = [i for i, w in enumerate(Tw) if w[3] == 9 + k]
            if len(ix) >= 2 and np.ptp(Jz[ix]) > 1e-3:
                slope = np.linalg.lstsq(np.c_[np.ones(len(ix)), Jz[ix]], rw[ix], rcond=None)[0][1]
                wall_turn_deg[wall[ix[0]]["id"]] = math.degrees(x[2] - slope)
    return dict(T_AP=T_AP, c_M=c_M, x=x, sd=np.sqrt(np.clip(np.diag(cov), 0, None)),
                rms_pad_mm=float(np.sqrt(np.mean((r[:n] * sig["pad_mm"]) ** 2))) if n else float("nan"), n_pad=n,
                rms_wall_mm=float(np.sqrt(np.mean((r[n:n + m] * sig["wall_mm"]) ** 2))) if m else float("nan"), n_wall=m,
                wall_offset_mm=x[9:], wall_turn_deg=wall_turn_deg, chi2=float(r @ r), dof=int(len(r) - nx),
                seat_change_mm_deg=pose_error(T_AP0, T_AP), probe_change_mm=float(np.linalg.norm(c_M - c_M0)),
                pad_ids=[t["id"] for t in pad], pad_resid_mm=(r[:n] * sig["pad_mm"]).tolist())


def fit_rod_registration(geo: Geometry, reg: Registration, touches, sig=REG_SIGMA_PROBE):
    """Probe mode with a rod (11-03): the rod and the pad's place from the rod's touches, the pad's place seeded from the 2
    mm sphere's session of the same seat (the tips stay seated through the tool change). Fitted: the pad's T_AP about the
    seed (6, prior seat_rot_deg and seat_t_mm), the rod's axis point c_M across the axis (2, prior 1 mm) and the axis
    turned about z_M (the tool's axis) and about the first direction square to it (2, priors 4 and 2 deg). A sphere's
    touches leave the mount's turn about the tool's axis and its error on the flange in T_AP, which suits the sphere's
    point but not the rod's line (in the simulator the seeded pad stood 1.6 mm and 0.7 deg off, and a fixed rod correction
    left 0.12 mm rms), so the pad is fitted again with the rod. A pad touch's residual is the pad surface's least distance
    from the rod's axis at the onset (geo.line_distance) less R. Returns fit_probe_registration's dict (x: the seat's 6,
    then c_M's change in M) with axis_M and axis_turn_deg."""
    pad = [t for t in touches if t["kind"] == "pad"]
    T_AP0, c_M0 = reg.T_AP, reg.c_M
    a_M0 = geo.axis_M
    e1 = np.cross(a_M0, [0.0, 0.0, 1.0] if abs(a_M0[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a_M0, e1)
    Ts = [np.array(t["T_BE"], float) @ reg.T_EM for t in pad]
    sd = np.r_[np.radians(sig["seat_rot_deg"]), sig["seat_t_mm"], [1.0, 1.0], np.radians([4.0, 2.0])]

    def unpack(x):
        a = rot(EZ, math.degrees(x[8])) @ rot(e1, math.degrees(x[9])) @ a_M0
        return perturb(T_AP0, x[0:6]), c_M0 + x[6] * e1 + x[7] * e2, a / np.linalg.norm(a)

    def resid(x):
        T_AP, c, a = unpack(x)
        T_PB = inv(reg.T_BI @ T_AP)
        r = [(geo.line_distance(apply(T_PB @ T, c), (T_PB @ T)[:3, :3] @ a) - geo.R) / sig["pad_mm"] for T in Ts]
        return np.r_[r, x / sd]

    x, r, cov = lm(resid, np.zeros(10))
    T_AP, c_M, a_M = unpack(x)
    n = len(pad)
    dc = c_M - c_M0
    sx = np.sqrt(np.clip(np.diag(cov), 0, None))
    return dict(T_AP=T_AP, c_M=c_M, axis_M=a_M, axis_turn_deg=np.degrees(x[8:10]), axis_turn_sd_deg=np.degrees(sx[8:10]),
                x=np.r_[x[0:6], dc], sd=np.r_[sx[0:6], np.abs(e1) * sx[6] + np.abs(e2) * sx[7]],
                rms_pad_mm=float(np.sqrt(np.mean((r[:n] * sig["pad_mm"]) ** 2))) if n else float("nan"), n_pad=n,
                rms_wall_mm=float("nan"), n_wall=0, wall_offset_mm=np.zeros(0), wall_turn_deg={}, chi2=float(r @ r),
                dof=int(len(r) - 10), seat_change_mm_deg=pose_error(T_AP0, T_AP), probe_change_mm=float(np.linalg.norm(dc)),
                pad_ids=[t["id"] for t in pad], pad_resid_mm=(r[:n] * sig["pad_mm"]).tolist())


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
    """Binary file of float64 rows (or of dtype rows), written by a thread of its own: write() only queues the bytes, and
    the thread writes them, flushes every 0.5 s and fsyncs every 5 s.

    The HEX-E's reader is the writer of ft.bin, and any wait in its thread is a gap in the stream. On 2026-10-22 the
    reader's own fsync blocked it 0.1 to 0.4 s on the lab computer's SSD, and later, with the fsync moved out, its write
    still waited 0.36 s while a 160 MB copy filled the page cache (dirty pages throttle every writer). Each such wait
    aborted an action with no HEX-E sample for 100 ms while the sensor sampled on."""

    def __init__(self, path, dtype=None):
        self.f = open(path, "ab")
        self.dtype = dtype
        self.lock = threading.Lock()
        self.q = queue.Queue()
        self.closed = False
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def write(self, rows):
        self.q.put(np.asarray(rows, "<f8" if self.dtype is None else self.dtype).tobytes())

    def _run(self):
        last_flush = last_sync = time.time()
        while True:
            try:
                b = self.q.get(timeout=0.5)
            except queue.Empty:
                b = None
            with self.lock:
                if self.f.closed:
                    return
                if b is not None:
                    self.f.write(b)
                now = time.time()
                if now - last_flush > 0.5:
                    self.f.flush()
                    last_flush = now
                if now - last_sync > 5.0:
                    os.fsync(self.f.fileno())
                    last_sync = now
            if self.closed and self.q.empty():
                return

    def close(self):
        self.closed = True
        self.th.join(timeout=10.0)
        with self.lock:
            while not self.q.empty():                            # anything the thread had not written yet
                self.f.write(self.q.get_nowait())
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


class HeldBoardStream(threading.Thread):
    """The boards of the tips on the holder (2026-10-27: two v3 tips beside the robot's, for the crosstalk maps) through
    reskin5x_plot.MuxReader (firmware/reskin5x_mux_stream, the boards behind an I2C mux): every board's records are
    drained every 0.2 s into held.bin (host time, the board's mux channel, 5 chips x [T, x, y, z]) and a ring per
    channel."""

    DTYPE = np.dtype([("t", "<f8"), ("b", "<u1"), ("r", "<f4", (20,))])

    def __init__(self, port, appender=None):
        super().__init__(daemon=True)
        sys.path.insert(0, str(HERE))
        import reskin5x_plot as RP
        self.reader = RP.MuxReader(port, keep=5.0)
        self.channels = list(self.reader.channels)
        self.rings = {c: Ring() for c in self.channels}
        self.n = {c: 0 for c in self.channels}
        self.app, self.stop = appender, False

    def run(self):
        self.reader.start()
        while not self.stop:
            time.sleep(0.2)
            rows = []
            for b in self.reader.boards:
                with b.lock:
                    recs, b.all = b.all, []
                for t, r in recs:
                    rows.append((t, b.channel, r.ravel()))
                    self.rings[b.channel].add(t, r.ravel())
                self.n[b.channel] += len(recs)
            if rows and self.app is not None:
                rows.sort(key=lambda x: x[0])
                out = np.zeros(len(rows), self.DTYPE)
                for k, row in enumerate(rows):
                    out[k] = row
                self.app.write(out)
        self.reader.stop = True


class URRobot:
    """ur_rtde on the CB3: flange poses (the controller's TCP set to zero), mm and 4 x 4 here, m and rotation vectors
    there. A thread logs t_host, t_controller, q (6), flange pose (6, m and rad), safety mode at 125 Hz."""

    def __init__(self, ip, appender=None, rate=125.0):
        import rtde_control
        import rtde_receive
        self.ip = ip
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

    def speed_fraction(self):
        """The controller's speed slider, 0 to 1 (2026-10-25: 0.30 since at least 10-22, set by no one here)."""
        return float(self.r.getTargetSpeedFraction())

    def set_speed_fraction(self, f):
        import rtde_io
        io = rtde_io.RTDEIOInterface(self.ip)
        try:
            return io.setSpeedSlider(float(f))
        finally:
            io.disconnect()

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
    def __init__(self, args, log, probe=False):
        if probe and not args.held_board:
            raise SystemExit("probe mode reads the seat's board among the held ones: give --held-board")
        self.ft = OptoForceStream(args.ft, rate=args.ft_rate, appender=log.appender("ft.bin"), stream=args.ft_stream,
                                  filt=args.ft_filter)
        self.ft.configure()
        self.ft.start()
        self.board = None                                   # probe mode: no tip on the robot unless --board names one
        if args.board or not probe:
            self.board = BoardStream(args.board, appender=log.appender("board.bin", BoardStream.DTYPE))
            self.board.start()
        self.held = None
        if args.held_board:
            if self.board is not None and os.path.realpath(args.held_board) == os.path.realpath(self.board.reader.port):
                raise SystemExit(f"--held-board {args.held_board} is the robot tip's board ({self.board.reader.port})")
            self.held = HeldBoardStream(args.held_board, appender=log.appender("held.bin", HeldBoardStream.DTYPE))
            self.held.start()
        self.robot = URRobot(args.robot, appender=log.appender("robot.bin"))

    def now(self):
        return time.time()

    def sleep(self, dt):
        time.sleep(dt)

    def operator(self, prompt, kind=None):
        return input(f"\n>>> {prompt} [Enter] ")

    def close(self):
        self.ft.stop = True
        if self.board is not None:
            self.board.stop = True
        if self.held is not None:
            self.held.stop = True
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
        ww = getattr(args, "wrist_window", None) or self.safety.get("wrist_window") or "30,100"
        self.wrist_window = tuple(float(v) for v in (ww.split(",") if isinstance(ww, str) else ww))
        tw = self.safety.get("travel_wrist_window")                 # probe mode: the lift and the home path run in
        self.travel_window = tuple(float(v) for v in tw) if tw else None   # this window, from the homing's branch
        self.wrist_travel = False
        self.safety.setdefault("abort_N", 10.0)
        self.safety.setdefault("abort_Nm", 1.5)
        self.safety.setdefault("box_mm", 400.0)
        self.touches, self.pivots = [], []
        self.n_abort = 0
        self.n_early = 0                # contacts before the slow approach in this run (EARLY_STOP)
        self.n_miss = 0                 # presses with no level reached in this run (MISS_STOP)
        self.dry = float(args.dry or 0.0)
        self.probe = self.geo.probe is not None
        self.last_d = None              # probe mode: the last target's push direction (base), for the recovery
        self.from_homing = False        # probe mode: the robot stands where the homing left it (out of the key)
        self.started = False            # probe mode with a home path: the session's first travel has been made
        self.prev_ok = None             # the last action that ended ok in this run
        ids = [a["id"] for a in plan["actions"]]
        self.prev_in_plan = dict(zip(ids[1:], ids[:-1]))
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
        held = getattr(self.be, "held", None)
        if self.probe and held is not None:                       # probe mode: the held tip's board on its mux channel
            t, v = held.rings[int(self.geo.seat["held_channel"])].since(t0, t1)
        else:
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

    def window(self):
        """Joint 5's window now: the plan's travel window while the lift and the home path run, else the work window."""
        return self.travel_window if self.wrist_travel and self.travel_window else self.wrist_window

    def wrist_ok(self, q):
        lo, hi = self.window()
        return lo <= math.degrees(float(q[4])) <= hi

    def check_wrist(self, q=None, target=False):
        """Abort when joint 5 leaves --wrist-window (deg): near 0 and 180 deg the wrist is singular, and past about 90
        deg the HEX-E's cable port can reach the forearm (2026-10-22: pressed at 173.5 deg, the owner stopped it)."""
        q = self.be.robot.q() if q is None else q
        if not self.wrist_ok(q):
            if not target:
                self.be.robot.stopL(3.0)
            lo, hi = self.window()
            raise Abort(f"joint 5 {'target' if target else 'at'} {math.degrees(float(q[4])):.1f} deg, outside "
                        f"{lo:.0f} to {hi:.0f} deg")

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

    def goto(self, T, v=0.02, a=0.2, joint=False, jv=0.4, ja=0.6, q_ref=None):
        if not self.in_box(T):
            raise Abort("target outside the box around the indenter")
        _, w0 = self.be.ft.ring.latest()
        if joint:                       # joint moves go to the IK branch the plan checked: q_ref (probe mode: the action's
            if q_ref is None:           # own joints, or the hub's), else the homing pose's
                q_ref = self.reg.q_home if getattr(self.reg, "q_home", None) is not None else self.be.robot.q()
            q = self.be.robot.ik(T, np.asarray(q_ref, float))
            if q is None:
                raise Abort("no IK solution")
            self.check_wrist(q, target=True)
            self.be.robot.moveJ(q, jv, ja, asynchronous=True)
        else:
            self.be.robot.moveL(T, v, a, asynchronous=True)
        self.await_move(w0, T=T)

    def goto_q(self, q_deg, jv=0.4, ja=0.6):
        """A joint waypoint of the plan's home path (deg): a joint move there, guarded as goto's."""
        q = np.radians(np.asarray(q_deg, float))
        self.check_wrist(q, target=True)
        _, w0 = self.be.ft.ring.latest()
        self.be.robot.moveJ(q, jv, ja, asynchronous=True)
        self.await_move(w0, q=q)

    def await_move(self, w0, T=None, q=None):
        """Watch a move started asynchronously until the robot is steady: abort on the HEX-E's or the wrist's checks, a
        load past travel_abort_N, 60 s, or a stop off its target (T, 1 mm and 0.5 deg; or joints q, 0.5 deg)."""
        def off():                      # mm (or deg of the farthest joint) from the target
            if q is not None:
                return math.degrees(float(np.abs(self.be.robot.q() - q).max()))
            return pose_error(self.be.robot.pose(), T)[0]

        t0 = self.now()
        self.sleep(0.05)
        lim = float(self.safety.get("travel_abort_N", 8.0))      # the payload's weight turns by at most 3.6 N
        while not self.be.robot.steady() or (self.now() - t0 < 0.2 and off() > (0.5 if q is not None else 1.0)):
            self.check_ft()
            _, w = self.be.ft.ring.latest()
            if w0 is not None and np.linalg.norm(w[:3] - w0[:3]) > lim:
                self.be.robot.stopL(3.0)
                raise Abort(f"load of {np.linalg.norm(w[:3] - w0[:3]):.1f} N while travelling")
            self.sleep(0.01)
            if self.now() - t0 > 60.0:
                raise Abort("travel took over 60 s")
        if q is not None:
            if off() > 0.5:
                raise Abort(f"the robot stopped {off():.2f} deg from its joint target")
            return
        e = pose_error(self.be.robot.pose(), T)
        if e[0] > 1.0 or e[1] > 0.5:
            raise Abort(f"the robot stopped {e[0]:.2f} mm, {e[1]:.2f} deg from its target")

    def T_BE_for(self, T_IP):
        return self.reg.T_BI @ T_IP @ inv(self.reg.T_EP())

    def T_BE_for_mount(self, T_IM):
        return self.reg.T_BI @ T_IM @ inv(self.reg.T_EM)

    def approach_dir(self):
        if self.probe and self.last_d is not None:                # probe mode: each press has its own direction
            return self.last_d
        return -(self.reg.T_BI[:3, :3] @ EZ)

    def lifted(self, T, h_mm):
        """T moved up (along z_I) until the pad's lowest free-surface point stands h_mm above the indenter's top (at
        least 10 mm up)."""
        I = self.reg.T_BI
        pts = apply(inv(I) @ T @ self.reg.T_EP(), self.geo.pts)
        top = (self.geo.c_I + self.geo.R * EZ)[2]
        dz = max(10.0, h_mm + top - float(pts[:, 2].min()))
        return Tm(T[:3, :3], T[:3, 3] + I[:3, :3] @ EZ * dz)

    def back_along(self, T, mm):
        """Probe mode: the flange pose moved mm back along the tool's axis z_M (toward the HEX-E, away from the pad)."""
        return Tm(T[:3, :3], T[:3, 3] - (T @ self.reg.T_EM)[:3, 2] * mm)

    def travel_probe(self, T_target, a):
        """Probe mode: the ways of the workstation's replay (scripts/ur5_inverted_rig.py travel_options), the one it
        checked for this action (a['travel_ws']): straight (back travel.probe.retract_mm along the stem, across, in), joint
        (back safe_mm, a joint move to safe_mm back from the target, in) or hub (the joint move by way of the hub pose over
        the seat). From the homing, out of the key along -x_M by a['homing_lift_mm'] first, then the joint move. A straight
        way only follows the action the replay had before it; any other start takes the joint move. Joint moves solve near
        the action's own joints (a['q_rest']) and the hub's (hub.q), in the branch the replay checked.

        A plan with travel.probe.home_path_q (the owner's press configuration of 2026-11-02, another IK branch than the
        homing's) goes from the homing's lift through those joints (deg; the last is the hub's) with joint 5 held to
        safety.travel_wrist_window, then on. The session's first travel starts from the homing key only when the robot
        stands on the line out of it; inside the work wrist window it backs off along the stem and goes by the hub; any
        other start stops the run."""
        cfg = dict(speed=1.0)
        cfg.update(self.plan.get("travel", {}))
        pr = cfg.get("probe", {})
        k = float(cfg["speed"])
        kj = min(k, 1.5)
        Tc = self.be.robot.pose()
        mm, deg = pose_error(Tc, T_target)
        if mm < 1e-3 and deg < 1e-3:
            return
        way = a.get("travel_ws", "joint")
        safe, ret = float(pr.get("safe_mm", 45.0)), float(pr.get("retract_mm", 12.0))
        vin = float(pr.get("in_v", 0.01))              # m/s: the last move in to the rest (11-03: 0.02 saves 1 to 7 s a travel)
        path, hub, q_rest = pr.get("home_path_q"), pr.get("hub") or {}, a.get("q_rest")
        if path and q_rest is None:
            raise Abort("a joint move with no q_rest in a plan with a home path")
        if path and not self.started:
            self.started = True
            if not self.from_homing:
                if self.on_homing_line(Tc, float(a.get("homing_lift_mm", 30.0))):
                    self.from_homing = True
                elif self.wrist_ok(self.be.robot.q()):
                    way = "hub"                                      # over the seats, placed by hand
                else:
                    raise Abort("the robot stands neither out of the homing key nor in the work wrist window: take it "
                                "to the key or over a seat by hand")
        if self.from_homing:
            lift_mm = float(a.get("homing_lift_mm", 30.0))
            self.wrist_travel = bool(path)               # the lift and the home path start in the homing's branch
            try:
                if path:                                 # to the line's end (12 mm of homing and the lift), from the seat
                    self.goto(self.T_lift(lift_mm), 0.02 * k, 0.2 * k)     # or from anywhere on the line
                    for q in path:
                        self.goto_q(q, 0.4 * kj, 0.6 * kj)
                else:
                    up = Tc[:3, :3] @ (self.reg.T_EM[:3, :3] @ np.array([-1.0, 0.0, 0.0]))
                    self.goto(Tm(Tc[:3, :3], Tc[:3, 3] + up * lift_mm), 0.02 * k, 0.2 * k)
            finally:
                self.wrist_travel = False
            self.from_homing = False
            way = "joint" if path else ("hub" if way == "hub" else "joint")    # the home path ends at the hub's joints
        elif way == "straight" and self.prev_ok == self.prev_in_plan.get(a["id"]) and deg <= 20.0 and mm <= 30.0:
            self.goto(self.back_along(Tc, ret), 0.02 * k, 0.2 * k)
            self.goto(self.back_along(T_target, ret), 0.02 * k, 0.2 * k)
            self.goto(T_target, vin * k, 0.2 * k)
            return
        else:
            way = "hub" if way == "hub" else "joint"
            if not self.clear_above(Tc, safe):          # back off only near the pad: an aborted action's next travel
                self.goto(self.back_along(Tc, safe), 0.03 * k, 0.2 * k)    # climbed safe_mm each time (11-02, p1dry)
        if way == "hub":
            self.goto(np.array(hub["T_BE"], float), joint=True, jv=0.4 * kj, ja=0.6 * kj, q_ref=hub.get("q"))
        self.goto(self.back_along(T_target, safe), joint=True, jv=0.4 * kj, ja=0.6 * kj, q_ref=q_rest)
        self.goto(T_target, vin * k, 0.2 * k)

    def stage_return(self):
        """Probe mode with a home path: back to the homing key's line by the plan's way back: safe_mm along the stem and
        the hub (from the work wrist window), the home path backwards with joint 5 held to the travel window, then the
        lift pose (T_lift, the homing's branch). The robot ends 12 mm and the lift out of the key."""
        cfg = dict(speed=1.0)
        cfg.update(self.plan.get("travel", {}))
        pr = cfg.get("probe", {})
        path, hub = pr.get("home_path_q"), pr.get("hub") or {}
        if not (self.probe and path and "T_BE" in hub and self.reg.T_BE_home is not None):
            raise SystemExit("return needs probe mode, travel.probe.home_path_q and hub, and the homing")
        k = float(cfg["speed"])
        kj = min(k, 1.5)
        lift_mm = float(pr.get("homing_lift_mm", 30.0))
        self.log.event(event="start", id="return", stage="return", kind="return")
        if self.on_homing_line(self.be.robot.pose(), lift_mm):
            print("return: the robot already stands out of the homing key", flush=True)
        else:
            if not self.wrist_ok(self.be.robot.q()):
                raise SystemExit("return: the robot is outside the work wrist window; take it back by hand")
            self.goto(self.back_along(self.be.robot.pose(), float(pr.get("safe_mm", 45.0))), 0.03 * k, 0.2 * k)
            self.goto(np.array(hub["T_BE"], float), joint=True, jv=0.4 * kj, ja=0.6 * kj, q_ref=hub.get("q"))
            self.wrist_travel = True
            try:
                for q in reversed(path[:-1]):              # the path's last waypoint is the hub's joints
                    self.goto_q(q, 0.4 * kj, 0.6 * kj)
                self.goto(self.T_lift(lift_mm), joint=True, jv=0.4 * kj, ja=0.6 * kj, q_ref=self.reg.q_home)
            finally:
                self.wrist_travel = False
        self.from_homing = True
        self.log.event(event="end", id="return", status="ok")

    def up_key(self):
        """Probe mode with a home path: out of the homing key along the key's pin, z_I (the fixture's frame; 2026-11-02:
        the probe base sits about 2.7 deg about its axis off the fork's T_EM, so -x_M ran 2 mm off the pin over 42 mm)."""
        return self.reg.T_BI[:3, 2]

    def clear_above(self, T, mm):
        """Probe mode: whether the sphere's centre at flange pose T stands mm or more over the seat's apex (z_I), clear
        of the pad by mm and the sphere's radius."""
        seat = getattr(self.geo, "seat", None) or {}
        if not self.probe or "apex_A" not in seat or self.reg.T_BI is None:
            return False
        c = apply(inv(self.reg.T_BI) @ T @ self.reg.T_EM, np.asarray(self.reg.c_M, float))
        return float(c[2]) >= float(seat["apex_A"][2]) + float(self.geo.R) + mm

    def T_lift(self, lift_mm):
        """Probe mode: the homing pose moved out of the key along its pin by the homing's 12 mm and lift_mm."""
        Th = self.reg.T_BE_home
        return Tm(Th[:3, :3], Th[:3, 3] + self.up_key() * (12.0 + lift_mm))

    def on_homing_line(self, T, lift_mm, tol_mm=2.0, tol_deg=1.0):
        """Whether the flange pose T stands on the line out of the homing key, from seated to T_lift (within tol)."""
        Th = self.reg.T_BE_home
        if Th is None:
            return False
        up = self.up_key()
        d = T[:3, 3] - Th[:3, 3]
        s = float(d @ up)
        _, e_deg = pose_error(T, Th)
        return (-tol_mm <= s <= 12.0 + lift_mm + tol_mm and float(np.linalg.norm(d - s * up)) <= tol_mm
                and e_deg <= tol_deg)

    def travel(self, T_target, a=None):
        """From wherever the tool is to T_target near the indenter, as the plan checked it: up 10 mm, across, down; or,
        when the tool turns more than travel.turn_deg or moves more than travel.far_mm, up until the pad is travel.lift_mm
        above the indenter's top, across (in joint space when the turn exceeds travel.joint_deg), down. Probe mode:
        travel_probe."""
        if self.probe:
            return self.travel_probe(T_target, a or {})
        cfg = dict(turn_deg=20.0, far_mm=60.0, lift_mm=60.0, joint_deg=45.0, speed=1.0)
        cfg.update(self.plan.get("travel", {}))
        k = float(cfg["speed"])             # travel.speed scales the moves' speeds and accelerations (joints at most 1.5x)
        kj = min(k, 1.5)
        Tc = self.be.robot.pose()
        mm, deg = pose_error(Tc, T_target)
        if mm < 1e-3 and deg < 1e-3:
            return
        if deg > cfg["turn_deg"] or mm > cfg["far_mm"]:
            self.goto(self.lifted(Tc, cfg["lift_mm"]), 0.03 * k, 0.2 * k)
            self.goto(self.lifted(T_target, cfg["lift_mm"]), 0.03 * k, 0.2 * k, joint=deg > cfg["joint_deg"], jv=0.4 * kj,
                      ja=0.6 * kj)
            self.goto(self.lifted(T_target, 0.0), 0.02 * k, 0.2 * k)
        else:
            up = self.reg.T_BI[:3, :3] @ EZ
            self.goto(Tm(Tc[:3, :3], Tc[:3, 3] + up * 10.0), 0.02 * k, 0.2 * k)
            self.goto(Tm(T_target[:3, :3], T_target[:3, 3] + up * 10.0), 0.02 * k, 0.2 * k)
        self.goto(T_target, 0.01 * k, 0.2 * k)

    def target_pose(self, a, standoff):
        """(flange pose at the action's rest, approach direction) for a plan action."""
        tg = a["target"]
        st = standoff + self.dry
        if self.probe:                  # the sphere's centre standoff back along d from where it meets p, the mount at R_PM
            T_BP = self.reg.T_BP()
            d_P = np.asarray(tg["d"], float) / np.linalg.norm(tg["d"])
            c_B = apply(T_BP, np.asarray(tg["p"], float) + self.geo.R * np.asarray(tg["n"], float) - standoff * d_P)
            R_BM = T_BP[:3, :3] @ np.asarray(tg["R_PM"], float)
            self.last_d = T_BP[:3, :3] @ d_P
            # --dry backs off along the stem, as the travels retreat: along the push, a wall's or a flank's dry rest ran
            # sideways at the seats' height, into key 1 or to the elbow's reach (the simulator, 10-30)
            return self.back_along(Tm(R_BM, c_B - R_BM @ self.reg.c_M) @ inv(self.reg.T_EM), self.dry), self.last_d
        if tg["frame"] == "M":
            R = rot(EZ, a.get("twist", 0.0)) @ self.geo.R_ref @ rot_from_to(tg["n"], [1.0, 0.0, 0.0])
            c = self.geo.contact_point_I(st, tg.get("y_rod", 0.0))
            T_IM = Tm(R, c - R @ np.asarray(tg["p"], float))
            return self.T_BE_for_mount(T_IM), self.approach_dir()
        T_IP = self.geo.T_IP(tg["p"], tg["n"], a.get("twist", 0.0), st, a.get("offset", (0.0, 0.0)), tg.get("y_rod", 0.0))
        return self.T_BE_for(T_IP), self.approach_dir()

    # ---------------------------------------------------------------- the press primitive
    def press(self, a, T_rest, d_B, levels=(), stop_N=0.5, max_mm=3.0, v=0.0005, hold_s=0.6, rest_s=0.5, keep=False,
              fast_mm=0.0, v_fast=0.005, on_level=None):
        """From T_rest along d_B at v (m/s) until the push passes each level (hold there), or stop_N, or max_mm; then
        back to T_rest (unless keep, which leaves the pad pressed). With fast_mm, the first fast_mm of free travel go at
        v_fast, and the slow approach (t_move0) starts there; a push over FAST_STOP_N on the way sends the tool back to
        T_rest, and the whole approach then runs at v. on_level(N, tare_v, rec), after each level's hold, may move the
        tool in contact if it leaves it at the stopped pose (a sweep's offsets and slides). Returns the record of times."""
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
        if fast_mm > 0.0:
            rec["fast"] = self.fast_approach(T_rest, d_B, fast_mm, v_fast, tare_v)
        t_start = self.now()
        rec["t_move0"] = t_start
        self.be.robot.moveL(T_end, v, 0.05, asynchronous=True)
        f_hist, t_last = deque(maxlen=PUSH_AVG["probe" if self.probe else "tip"]), None
        while True:
            self.sleep(0.004)
            self.check_ft(tare_v)
            t, val = self.be.ft.ring.latest()
            if not self.probe or t != t_last:       # probe mode: each HEX-E sample once (this loop runs at 250 Hz), and
                f_hist.append(self.push_force(val, tare_v, d_B, R_BE))   # no level until the mean holds its samples
                t_last = t
            f = float(np.mean(f_hist))
            if k < len(lv) and (not self.probe or len(f_hist) == f_hist.maxlen) and f >= lv[k]:
                self.be.robot.stopL(0.05)
                self.wait_steady()
                if k == len(lv) - 1:
                    rec["t_stop"] = self.now()
                    if hold_s and any(abs(lv[k] - float(x)) < 1e-9 for x in levels):
                        t0 = self.now()
                        self.hold(hold_s, tare_v)
                        rec["levels"].append(dict(N=lv[k], t=[t0, self.now()]))
                        if on_level is not None:
                            on_level(lv[k], tare_v, rec)
                    break
                t0 = self.now()
                self.hold(hold_s, tare_v)
                rec["levels"].append(dict(N=lv[k], t=[t0, self.now()]))
                if on_level is not None:
                    on_level(lv[k], tare_v, rec)
                    f_hist.clear()                  # the push moved during the sweep: the next level from new samples
                k += 1
                self.be.robot.moveL(T_end, v, 0.05, asynchronous=True)
                continue
            if self.be.robot.steady() and self.now() - t_start > 0.3:
                rec["t_stop"] = self.now()
                rec["reached_max"] = True
                if on_level is not None and k < len(lv) and any(abs(lv[k] - float(x)) < 1e-9 for x in levels):
                    rec["at_depth_N"] = lv[k]           # a sweep's level not reached at max_depth_mm: its steps here
                    on_level(lv[k], tare_v, rec)        # (the workstation, 11-03 13:37), the push the HEX-E's
                break
        rec["T_stop"] = self.be.robot.pose()
        if keep:
            rec["tare_v"] = tare_v
            return rec
        # travel.back_v (m/s, before the slider): the return to the rest, 0.002 by default (11-03: 0.01 saves about 2.7 s
        # a press at the 0.30 slider, the workstation's timing of s1v6b, s2v5 and s3v4)
        self.be.robot.moveL(T_rest, float(self.plan.get("travel", {}).get("back_v", 0.002)), 0.05)
        t0 = self.now()
        self.mean_ft(rest_s)
        rec["t_rest1"] = [t0, self.now()]
        return rec

    def fast_approach(self, T_rest, d_B, fast_mm, v_fast, tare_v):
        """The free travel from T_rest to fast_mm along d_B at v_fast (m/s), watching the push (2026-10-25: the touch-offs
        crossed 5 mm of air at 0.05 mm/s, two minutes each). Returns the record."""
        R_BE = T_rest[:3, :3]
        T_f = Tm(R_BE, T_rest[:3, 3] + d_B * fast_mm)
        out = dict(mm=float(fast_mm), v=float(v_fast), t=[self.now(), None], contact=False)
        f_hist, t_last = deque(maxlen=PUSH_AVG["probe" if self.probe else "tip"]), None
        self.be.robot.moveL(T_f, v_fast, 0.1, asynchronous=True)
        t0 = self.now()
        while True:
            self.sleep(0.004)
            self.check_ft(tare_v)
            t, val = self.be.ft.ring.latest()
            if not self.probe or t != t_last:       # probe mode: as in press
                f_hist.append(self.push_force(val, tare_v, d_B, R_BE))
                t_last = t
            if (not self.probe or len(f_hist) == f_hist.maxlen) and float(np.mean(f_hist)) > FAST_STOP_N:
                self.be.robot.stopL(0.5)
                self.wait_steady()
                self.be.robot.moveL(T_rest, 0.002, 0.05)
                out["contact"] = True
                print(f"  contact before the slow approach ({float(np.mean(f_hist)):.2f} N): back to the rest pose, "
                      "the approach at its slow speed", flush=True)
                self.n_early += 1
                if self.n_early >= EARLY_STOP:
                    raise SystemExit(f"{self.n_early} contacts before the slow approach in this run: the registration or "
                                     "the frame may be off, or the tool meets something else; stopping at the rest pose")
                break
            # the CB3 settles about 0.03 mm from T_f and reports its pose with 0.025 mm of jitter: at 0.02 mm roll-0-tv004
            # waited out the timeout (2026-10-25 14:33). As in goto and the turns, steady after 0.2 s is arrived
            if self.be.robot.steady() and (self.now() - t0 > 0.2 or pose_error(self.be.robot.pose(), T_f)[0] < 0.1):
                break
            if self.now() - t0 > 10.0 + fast_mm / (v_fast * 1e3):
                self.be.robot.stopL(0.5)
                raise Abort("the fast approach did not end")
        self.wait_steady()
        out["t"][1] = self.now()
        return out

    def fast_mm(self, a, standoff, kind):
        """(mm of the standoff crossed fast, its speed m/s) from the plan's travel block: slow_mm_touch or slow_mm_press
        is the distance before the expected surface where the slow approach starts. Without them, (0, -): the whole
        standoff at the slow speed, as before 2026-10-25."""
        cfg = self.plan.get("travel", {})
        slow = cfg.get("slow_mm_touch" if kind == "touch" else "slow_mm_press")
        if slow is None:
            return 0.0, 0.005
        return max(0.0, float(standoff) - float(slow)), float(cfg.get("fast_v", 0.005))

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
            tv = float(a.get("turn_v", self.plan.get("travel", {}).get("turn_v", 0.004)))   # the action's own, else the plan's
            self.be.robot.moveL_path(path, tv, 0.05, 0.0002, asynchronous=True)   # on the CB3 the turn's rad/s: 0.004 = 0.23 deg/s
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
        self.check_slider()
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
            self.prev_ok = None
            self.recover()
            return None
        self.n_abort = 0
        self.log.event(event="end", id=a["id"], status="ok", rec=rec)
        self.done.add(a["id"])
        self.prev_ok = a["id"]
        return rec

    def recover(self):
        """After an abort: wait for the operator if the robot is stopped, then lift 5 mm away from the indenter."""
        if self.be.robot.stopped():
            self.be.operator("the robot is stopped: clear the protective stop on the pendant", kind="clear")
        try:
            Tc = self.be.robot.pose()
            if self.clear_above(Tc, 20.0):                       # probe mode: already clear of the pad (an abort before
                return                                           # any motion), so no lift
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
        held = getattr(self.be, "held", None)
        if held is not None:
            out["held_hz"] = {c: len(held.rings[c].since(t1 - 2.0)[0]) / 2.0 for c in held.channels}
            print(f"  held boards on {held.reader.port}: "
                  + ", ".join(f"mux channel {c} {r:.0f} records/s" for c, r in out["held_hz"].items()))
            if not out["held_hz"] or min(out["held_hz"].values()) < 10:
                raise SystemExit("a held tip's board is missing or slow: check the holder's boards and their USB")
            if self.probe and int(self.geo.seat["held_channel"]) not in held.channels:
                raise SystemExit(f"the seat's board (mux channel {self.geo.seat['held_channel']}) is not among the held "
                                 f"boards ({held.channels}): check geometry.seat.held_channel and the seat's cable")
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
        board_min = 10 if self.probe and held is not None else 50                # a held board shares its mux
        if out["ft_hz"] < 0.8 * self.args.ft_rate or out["board_hz"] < board_min:   # a 100 Hz poll counts 99.5-100.5
            raise SystemExit("a stream is missing or slow: check the HEX-E's DAQ (IP, power) and the board's USB")
        self.log.event(event="end", id="preflight", status="ok", rec=out)

    def home_from_file(self):
        """The flange pose of a recorded seating on the indenter's key (--home-from FILE:LABEL, a JSON line with label and
        T_tcp_mm, the flange pose in mm as 4 x 4: the hand seatings of 2026-10-22), for runs with nobody at the robot; and
        its joints (q_deg, rad here) when the record has them, else None."""
        path, _, label = self.args.home_from.rpartition(":")
        recs = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
        hit = [r for r in recs if r.get("label") == label]
        if not hit:
            raise SystemExit(f"--home-from: no record labelled {label!r} in {path}")
        q = hit[-1].get("q_deg")
        return np.array(hit[-1]["T_tcp_mm"], float), (np.radians(np.asarray(q, float)) if q is not None else None)

    def stage_homing(self):
        if "homing" in self.done and self.reg.T_BI is not None:
            return
        n = int(self.plan.get("homing", {}).get("repeats", 3))
        poses, q_seed = [], None
        self.log.event(event="start", id="homing", stage="homing", kind="homing")
        if getattr(self.args, "home_from", None):
            T, q_seed = self.home_from_file()               # the seating's own joints: the homing's branch, wherever
            poses = [T]                                     # the robot stands now
            n = 0
            print(f"homing: from the recorded seating {self.args.home_from} (no freedrive, nothing moves)", flush=True)
        what = (f"seat the probe's socket on the strip's key {int(self.plan.get('homing', {}).get('key', 0)) + 1} (palmar "
                "face down, the tool along the strip's +x)" if self.probe else
                "seat the mount's socket on the indenter's key (palmar face down, the finger along the indenter's +x)")
        for k in range(n):
            self.be.robot.teach(True)
            self.be.operator(f"homing {k + 1}/{n}: in freedrive, {what}, hold it seated", kind="seat")
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
        self.reg.q_home = self.be.robot.ik(Th, q_seed if q_seed is not None else self.be.robot.q())
        self.reg.T_BI = Th @ self.reg.T_EM @ inv(self.geo.T_IM_home)
        if getattr(self.args, "frame", None):       # the fixture's frame measured otherwise (both keys of the strip):
            T_fr = np.array(json.loads(Path(self.args.frame).read_text())["T_BA"], float)   # the seating still gives
            e = pose_error(T_fr, self.reg.T_BI)                                               # the homing's line
            print(f"homing: the fixture's frame from {self.args.frame}, {e[0]:.2f} mm and {e[1]:.2f} deg from the "
                  "seating's own", flush=True)
            self.reg.notes["frame_from"] = dict(file=str(self.args.frame), d_mm=e[0], d_deg=e[1])
            self.reg.T_BI = T_fr
        self.reg.notes["homing_spread_mm_deg"] = spread
        print(f"homing: {len(poses)} seatings within {max(s[0] for s in spread):.3f} mm, {max(s[1] for s in spread):.3f} deg "
              "of their mean")
        self.check_placement()
        self.save_reg()
        self.log.event(event="end", id="homing", status="ok", rec=dict(T_BE_home=Th, poses=poses, spread=spread))
        self.done.add("homing")
        self.from_homing = self.probe and n > 0               # the robot stands 12 mm out of the key

    def check_placement(self, max_mm=20.0, max_yaw=5.0, max_tilt=2.0):
        """The plan's reach and clearance checks hold for the placement it was built on (checks.placement_mm_deg: x, y, z
        in the base frame, yaw of x_I about z_B). Homing measures the real one; past the limits the operator moves the
        plywood, rebuilds the plan, or types 'continue'. A plan built on a measured frame (checks.T_BA, the probe plans of
        2026-11-02 on a tilted strip) is compared with that frame, its tilt with the frame's."""
        chk = self.plan.get("checks", {})
        pl, Tp = chk.get("placement_mm_deg"), chk.get("T_BA")
        if pl is None and Tp is None:
            return
        p, R = self.reg.T_BI[:3, 3], self.reg.T_BI[:3, :3]
        yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0))))
        if Tp is not None:
            Tp = np.array(Tp, float)
            pl = list(Tp[:3, 3]) + [math.degrees(math.atan2(Tp[1, 0], Tp[0, 0]))]
            tilt = math.degrees(math.acos(float(np.clip(Tp[:3, 2] @ R[:, 2], -1.0, 1.0))))
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
        self.travel(T_rest, a)
        fm, fv = self.fast_mm(a, st, "touch")
        rec = self.press(a, T_rest, d, levels=(), stop_N=float(a.get("stop_N", 0.5)),
                         max_mm=st + float(a.get("beyond_mm", 1.2)), v=float(a.get("v", 0.00015)), hold_s=0.0, rest_s=0.5,
                         fast_mm=fm, v_fast=fv)
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
                       p_set=(1.0,) if kind in ("mount", "wall") else (1.0, 1.25, 1.5, 1.75, 2.0))
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
        if self.probe:
            fit_fn = fit_rod_registration if self.geo.kind == "rod" else fit_probe_registration
            use, dropped = list(self.touches), []
            while True:                                 # a touch that met something else (REG_DROP_MM), the worst first
                fit = fit_fn(self.geo, self.reg, use)
                res = np.abs(fit["pad_resid_mm"])
                if not len(res) or res.max() <= REG_DROP_MM or len(dropped) >= REG_DROP_MAX:
                    break
                k = int(np.argmax(res))
                dropped.append((fit["pad_ids"][k], fit["pad_resid_mm"][k]))
                use = [t for t in use if not (t["kind"] == "pad" and t["id"] == fit["pad_ids"][k])]
            for i, r_ in dropped:
                print(f"register: {i} left out, {r_:+.2f} mm off the pad under the fit that held it (it met something "
                      "other than the pad)", flush=True)
            fit["dropped"] = [list(d) for d in dropped]
            if fit["rms_pad_mm"] > REG_RMS_MAX_MM:
                self.log.event(event="fit", id="register_probe_refused", rec=fit)
                raise SystemExit(f"register: the pad touches stand {fit['rms_pad_mm']:.3f} mm rms ({fit['n_pad']}; "
                                 f"{len(dropped)} left out), over {REG_RMS_MAX_MM} mm: the registration is kept as it was "
                                 "and the session stops before the grid")
            self.reg.T_AP, self.reg.c_M = fit["T_AP"], fit["c_M"]
            self.reg.notes["register_probe"] = {k: fit[k] for k in ("rms_pad_mm", "n_pad", "rms_wall_mm", "n_wall",
                                                                     "wall_offset_mm", "wall_turn_deg",
                                                                     "seat_change_mm_deg", "probe_change_mm", "sd", "x",
                                                                     "axis_M", "axis_turn_deg", "axis_turn_sd_deg")
                                                if k in fit}
            wt = fit["wall_turn_deg"]
            walls = (f", walls {fit['rms_wall_mm']:.3f} mm rms ({fit['n_wall']}; their offsets "
                     f"{np.round(fit['wall_offset_mm'], 3).tolist()} mm; each wall alone turns it "
                     + ", ".join(f"{k} {v:+.2f}" for k, v in wt.items()) + " deg)" if fit["n_wall"] else "")
            if len(wt) >= 2 and np.ptp(list(wt.values())) > 0.8:     # 2 sd at 0.02 mm a touch; e.g. a low touch on
                #                                                          the station block's face, 0.12 mm early: 1.5 deg
                print(f"WARNING the walls' turns differ by {np.ptp(list(wt.values())):.2f} deg: one of them may have met "
                      "something other than the tip's wall (the station block's face beside the low touch)", flush=True)
            if self.geo.kind == "rod":
                self.reg.axis_M = fit["axis_M"]                  # the rod's fitted axis (M), kept in registration.json
                print(f"register (probe, rod): pad touches {fit['rms_pad_mm']:.3f} mm rms ({fit['n_pad']}); the pad "
                      f"{fit['seat_change_mm_deg'][0]:.3f} mm, {fit['seat_change_mm_deg'][1]:.3f} deg from the seed; the rod's "
                      f"axis moved {fit['probe_change_mm']:.3f} mm across itself, c_M at "
                      f"{np.round(fit['c_M'], 3)} in M (sd {np.round(fit['sd'][6:9], 3)} mm), turned "
                      f"{fit['axis_turn_deg'][0]:+.2f} deg about z_M and {fit['axis_turn_deg'][1]:+.2f} deg across (sd "
                      f"{np.round(fit['axis_turn_sd_deg'], 2)})")
            else:
                print(f"register (probe): pad touches {fit['rms_pad_mm']:.3f} mm rms ({fit['n_pad']}){walls}; the seat's pad "
                      f"stands {fit['seat_change_mm_deg'][0]:.3f} mm, {fit['seat_change_mm_deg'][1]:.3f} deg from its CAD (the "
                      f"turn about z_P {math.degrees(fit['x'][2]):+.3f} deg, sd {math.degrees(fit['sd'][2]):.3f}); the sphere's "
                      f"centre {fit['probe_change_mm']:.3f} mm, at {np.round(fit['c_M'], 3)} in M (sd "
                      f"{np.round(fit['sd'][6:9], 3)} mm)")
            self.save_reg()
            self.log.event(event="fit", id="register_probe", rec=fit)
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
        lvl = float(self.plan.get("pivot", {}).get("level_N", 1.5))
        for a in acts:
            def one(a):
                T_rest, d = self.target_pose(a, 1.5)
                self.travel(T_rest)
                r = self.press(a, T_rest, d, levels=a.get("levels", (1.5, 3.0)), stop_N=a.get("stop_N", 3.0),
                               max_mm=4.0, v=0.0004, hold_s=float(a.get("hold_s", 1.5)))
                r["T_rest"] = T_rest
                # the board's change at lvl now, while the press's samples are in the streams' rings (180 s): read
                # after all six pivots, the reference's had left them at 30 % speed (2026-10-22, a crash)
                bl = self.board_level(r, lvl)
                r["board_level"] = None if bl is None else [np.asarray(bl[0]).tolist(), np.asarray(bl[1]).tolist()]
                return r
            r = self.act(a, one)
            if r is not None:
                recs[a["id"]] = (a, r)
        if self.dry:
            return
        def stored(r):
            bl = r.get("board_level")
            return None if bl is None else (np.array(bl[0], float), np.array(bl[1], float))

        ref = [v for v in recs.values() if v[0].get("role") == "ref"]
        offs = [v for v in recs.values() if v[0].get("role") == "offset"]
        turns = [v for v in recs.values() if v[0].get("role") == "turn"]
        if not ref or len(offs) < 4:
            print("pivot: missing presses")
            return
        bl0 = stored(ref[0][1])
        if bl0 is None:
            print("pivot: the reference press held no level; no check")
            return
        b0, f0 = bl0
        cols, rows = [], []
        for a, r in offs:
            bl = stored(r)
            if bl is not None:
                rows.append(np.array(a["offset_P"]))
                cols.append(bl[0] - b0)
        J, *_ = np.linalg.lstsq(np.array(rows), np.array(cols), rcond=None)       # (2, 15): change per mm along t1, t2
        J = J.T
        out = []
        t1, t2 = (np.array(self.plan["pivot"][k], float) for k in ("t1", "t2"))
        for a, r in turns:
            bl = stored(r)
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
        self.travel(T_rest, a)
        fm, fv = self.fast_mm(a, float(a.get("standoff", 1.5)), "press")
        r = self.press(a, T_rest, d, levels=a.get("levels", (0.25, 0.5, 1.0, 2.0)), stop_N=a.get("stop_N", max(a.get("levels", [2.0]))),
                       max_mm=float(a.get("standoff", 1.5)) + float(a.get("max_depth_mm", 2.5)), v=float(a.get("v", 0.0005)),
                       hold_s=float(a.get("hold_s", 0.6)), fast_mm=fm, v_fast=fv)
        r["T_rest"] = T_rest
        return r

    def grid_sweep(self, a):
        """Probe mode, the owner's cone of contact (11-03): a press whose level stops each carry offsets, and whose level
        at_N carries slides. After each level's hold, for each [r_mm, az_deg] of that level (a['offsets'][k], aligned with
        a['levels']), the flange moves r along the pad's tangent plane at azimuth az (t1 = target.t1, t2 = n x t1, in P;
        the orientation kept) at v_offset, settles settle_s and holds step_hold_s, then goes back to the stopped pose
        (via_centre) before the next. Each slide {az_deg, mm, at_N} goes mm out at v_slide, holds step_hold_s and comes
        back. Every move and hold watches the load as the press does and the push across the pad's normal against
        ft_t_cap_N. The press then goes on to its next level. Recorded: offsets [{N, r_mm, az_deg, t, T_BE}] and slides
        [{az_deg, mm, at_N, t_out, t_hold, t_back, T_end}]."""
        if not self.probe:
            raise Abort("a sweep is the probe mode's")
        st = float(a.get("standoff", 1.5))
        T_rest, d = self.target_pose(a, st)
        tg = a["target"]
        if "t1" not in tg:
            raise Abort("a sweep needs target.t1")
        R_BP = self.reg.T_BP()[:3, :3]
        n_P = np.asarray(tg["n"], float) / np.linalg.norm(tg["n"])
        t1_P = np.asarray(tg["t1"], float)
        t1_P = (t1_P - (t1_P @ n_P) * n_P) / np.linalg.norm(t1_P - (t1_P @ n_P) * n_P)
        t1_B, t2_B, n_B = R_BP @ t1_P, R_BP @ np.cross(n_P, t1_P), R_BP @ n_P
        levels = [float(x) for x in a.get("levels", (0.5, 1.0, 2.0))]
        offsets = a.get("offsets") or []
        slides = a.get("slides") or []
        # the push across the pad's normal: at most ft_t_cap_N, and at most ft_t_ratio x the push along it + ft_t_floor_N
        # (friction gives up to mu x the normal push; the simulator's mu is 1.0, the TPU's 0.37 to 0.44 on PLA)
        cap = float(a.get("ft_t_cap_N", 4.0))
        ratio, floor = float(a.get("ft_t_ratio", 1.2)), float(a.get("ft_t_floor_N", 0.5))
        v_off, v_sl = float(a.get("v_offset", 0.002)), float(a.get("v_slide", 0.0005))
        settle, hold = float(a.get("settle_s", 0.2)), float(a.get("step_hold_s", 0.2))
        via = bool(a.get("via_centre", True))
        last = [0.0, 0.0]                               # the last push along and across the pad's normal (N)
        dbg = bool(os.environ.get("SWEEP_DEBUG"))

        class Skip(Exception):
            pass

        def watch(tare_v, R_BE, on_ratio):
            """The load as the press watches it, the push across against cap (always an abort), and against ratio x the
            push along + floor: on_ratio "abort", "skip" (raises Skip: the offsets of 11-03's face-717 in the simulator,
            where a light contact unloaded) or "off" (the way back to the stopped pose after a skip)."""
            self.check_ft(tare_v)
            _, v = self.be.ft.ring.latest()
            f_B = R_BE @ np.linalg.solve(self.reg.K, np.asarray(v[:3], float) - tare_v[:3])
            f_n = float(f_B @ n_B)
            f_t = float(np.linalg.norm(f_B - f_n * n_B))
            last[:] = [f_n, f_t]
            over = f"the push across the pad {f_t:.2f} N at {f_n:.2f} N along its normal"
            if f_t > cap:
                self.be.robot.stopL(3.0)
                raise Abort(f"{over}, over the sweep's {cap:g} N")
            if on_ratio != "off" and f_t > ratio * max(f_n, 0.0) + floor:     # past any friction (mu 0.4 to 1)
                self.be.robot.stopL(3.0)
                msg = f"{over}, over {ratio:g} x the push along + {floor:g} N"
                raise Skip(msg) if on_ratio == "skip" else Abort(msg)

        def go(T, v, tare_v, on_ratio):
            self.be.robot.moveL(T, v, 0.05, asynchronous=True)
            t_m = self.now()
            self.sleep(0.02)
            while not self.be.robot.steady() or self.now() - t_m < 0.05:
                watch(tare_v, T[:3, :3], on_ratio)
                self.sleep(0.004)
                if self.now() - t_m > 30.0:
                    self.be.robot.stopL(3.0)
                    raise Abort("a sweep's move did not end")

        def stay(dur, tare_v, R_BE, on_ratio):
            t0 = self.now()
            while self.now() - t0 < dur:
                watch(tare_v, R_BE, on_ratio)
                self.sleep(0.01)
            return [t0, self.now()]

        def on_level(N, tare_v, rec):
            T0 = self.be.robot.pose()
            R_BE = T0[:3, :3]
            rec.setdefault("offsets", [])
            rec.setdefault("slides", [])
            rec.setdefault("skips", [])
            k = next((j for j, x in enumerate(levels) if abs(x - N) < 1e-9), None)
            for r_mm, az in (offsets[k] if k is not None and k < len(offsets) else []):
                u = math.cos(math.radians(az)) * t1_B + math.sin(math.radians(az)) * t2_B
                T = Tm(R_BE, T0[:3, 3] + u * float(r_mm))
                if dbg:
                    print(f"    {a['id']} {N:g} N: before r {r_mm} az {az}: along {last[0]:.2f} across {last[1]:.2f}", flush=True)
                try:
                    go(T, v_off, tare_v, "skip")
                    stay(settle, tare_v, R_BE, "skip")
                    tw = stay(hold, tare_v, R_BE, "skip")
                    if dbg:
                        print(f"    {a['id']} {N:g} N: at r {r_mm} az {az}: along {last[0]:.2f} across {last[1]:.2f}", flush=True)
                    rec["offsets"].append(dict(N=N, r_mm=float(r_mm), az_deg=float(az), t=tw, T_BE=self.be.robot.pose(),
                                               at_depth=rec.get("at_depth_N") == N))
                    if via:
                        go(T0, v_off, tare_v, "skip")
                except Skip as e:                       # the rest of this level's offsets left out, the press goes on
                    self.wait_steady()
                    rec["skips"].append(dict(N=N, r_mm=float(r_mm), az_deg=float(az), t=self.now(), why=str(e)))
                    print(f"  {a['id']}: {N:g} N's offsets stop at r {r_mm} az {az:g}: {e}", flush=True)
                    go(T0, v_off, tare_v, "off")
                    break
            if not via:
                go(T0, v_off, tare_v, "skip")
            for s in slides:
                if abs(float(s["at_N"]) - N) > 1e-9:
                    continue
                u = math.cos(math.radians(s["az_deg"])) * t1_B + math.sin(math.radians(s["az_deg"])) * t2_B
                T = Tm(R_BE, T0[:3, 3] + u * float(s["mm"]))
                try:
                    t0 = self.now()
                    go(T, v_sl, tare_v, "skip")
                    t_out = [t0, self.now()]
                    tw = stay(hold, tare_v, R_BE, "skip")
                    T_end = self.be.robot.pose()
                    t0 = self.now()
                    go(T0, v_sl, tare_v, "skip")
                except Skip as e:                       # the rest of this level's slides left out (in the simulator at
                    self.wait_steady()                  # mu 1.0 a slide's push along fell to 0.85 N at 2 N)
                    rec["skips"].append(dict(N=N, slide_az_deg=float(s["az_deg"]), t=self.now(), why=str(e)))
                    print(f"  {a['id']}: {N:g} N's slides stop at az {s['az_deg']:g}: {e}", flush=True)
                    go(T0, v_sl, tare_v, "off")
                    break
                rec["slides"].append(dict(az_deg=float(s["az_deg"]), mm=float(s["mm"]), at_N=N, t_out=t_out, t_hold=tw,
                                          t_back=[t0, self.now()], T_end=T_end))

        self.travel(T_rest, a)
        fm, fv = self.fast_mm(a, st, "press")
        r = self.press(a, T_rest, d, levels=levels, stop_N=a.get("stop_N", max(levels)),
                       max_mm=st + float(a.get("max_depth_mm", 2.5)), v=float(a.get("v", 0.0005)),
                       hold_s=float(a.get("hold_s", 0.6)), fast_mm=fm, v_fast=fv, on_level=on_level)
        r["T_rest"] = T_rest
        return r

    def stage_presses(self, stage):
        acts = self.actions(stage)
        for k, a in enumerate(acts):
            if a["kind"] == "press":
                r = self.act(a, self.grid_press)
            elif a["kind"] == "sweep":
                r = self.act(a, self.grid_sweep)
            elif a["kind"] in ("slide", "roll", "twist"):
                r = self.act(a, self.contact_turns)
            else:
                continue
            if r is not None and r.get("levels"):
                top = r["levels"][-1]["N"]
                print(f"  {a['id']} ({k + 1}/{len(acts)}): to {top:.2f} N", flush=True)
            elif r is not None:                         # its whole depth with no level reached: the pad stood elsewhere
                self.n_miss += 1
                print(f"  {a['id']} ({k + 1}/{len(acts)}): no level reached", flush=True)
                if self.n_miss >= MISS_STOP:
                    raise SystemExit(f"{self.n_miss} presses reached no level in this run: the registration or the frame "
                                     "may be off; stopping")

    def contact_turns(self, a):
        """Press to a['press_N'] and hold, then the moves in turn, each back to the pressed pose at its end: slide =
        turns about the sphere's centre (the contact slides along dir_deg in I at constant indentation, R deg / 57.3 mm
        per degree), roll = the same axis through the contact point (the contact rolls), twist = about the vertical
        through the contact (torsion). Probe mode: twist only, the sphere turned about the pad's normal at the target
        through its own centre, as the workstation's replay turned it (at most standoff + max_depth_mm along the push)."""
        st = float(a.get("standoff", 1.5))
        T_rest, d = self.target_pose(a, st)
        if self.probe and a["kind"] != "twist" and not (a["kind"] == "roll" and self.geo.kind == "rod"):
            raise Abort(f"a {a['kind']} is the tip mode's (probe mode: twists, and rolls of a rod)")
        self.travel(T_rest, a)
        fm, fv = self.fast_mm(a, st, "press")
        r = self.press(a, T_rest, d, levels=(a["press_N"],), stop_N=a["press_N"],
                       max_mm=st + (float(a.get("max_depth_mm", 1.0)) if self.probe else 3.0), v=0.0004, hold_s=0.5,
                       keep=True, fast_mm=fm, v_fast=fv)
        if self.probe:
            n_B = self.reg.T_BP()[:3, :3] @ np.asarray(a["target"]["n"], float)
        else:
            I = self.reg.T_BI
            c_B = apply(I, self.geo.c_I)
            top_B = apply(I, self.geo.c_I + self.geo.R * EZ)
        r["moves"] = []
        try:
            for mv in a["moves"]:
                ang = math.radians(mv.get("dir_deg", 0.0))
                u_I = np.array([math.cos(ang), math.sin(ang), 0.0])
                if self.probe and a["kind"] == "roll":       # the rod rolls on the pad (11-03): about its contact line, along
                    T_BM = np.array(r["T_stop"]) @ self.reg.T_EM  # its fitted axis, R from its axis along the pad's normal
                    a_M = np.asarray(getattr(self.reg, "axis_M", self.geo.axis_M), float)
                    centre = apply(T_BM, self.reg.c_M) - self.geo.R * n_B / np.linalg.norm(n_B)
                    axis = T_BM[:3, :3] @ (a_M / np.linalg.norm(a_M))
                elif self.probe:
                    centre, axis = apply(np.array(r["T_stop"]) @ self.reg.T_EM, self.reg.c_M), n_B / np.linalg.norm(n_B)
                elif a["kind"] == "slide":
                    centre, axis = c_B, I[:3, :3] @ np.cross(EZ, u_I)
                elif a["kind"] == "roll":
                    centre, axis = top_B, I[:3, :3] @ np.cross(EZ, u_I)
                else:
                    centre, axis = top_B, I[:3, :3] @ EZ
                self.run_turns(a, r, centre, axis, list(mv["degs"]) + [0.0], hold_s=float(a.get("hold_s", 0.4)))
                r["moves"].append(dict(dir_deg=mv.get("dir_deg", 0.0), centre_B=centre, axis_B=axis, turns=r.pop("turns")))
        finally:                                        # back along d to 0.5 mm behind the rest (11-03: st + 0.5 from the
            T_now = self.be.robot.pose()                # stop left a press that ran to its full depth on the pad)
            back = max(float((T_now[:3, 3] - T_rest[:3, 3]) @ d), 0.0) + 0.5
            self.be.robot.moveL(Tm(T_now[:3, :3], T_now[:3, 3] - d * back), 0.002, 0.05)
        t0 = self.now()
        self.mean_ft(0.5)
        r["t_rest1"] = [t0, self.now()]
        r["T_rest"] = T_rest
        return r

    def stage_hover(self):
        acts = self.actions("hover")
        for k, a in enumerate(acts):
            if a["kind"] != "hover":
                continue
            r = self.act(a, self.hover)
            if r is not None:
                print(f"  {a['id']} ({k + 1}/{len(acts)}): held {a.get('dwell_s', 2.0):g} s, "
                      f"|F| at most {r['peak_N']:.2f} N", flush=True)

    def hover(self, a):
        """No contact (2026-10-27, the crosstalk maps with the tips on the holder): the pad's target point (a['target'],
        P frame) held at a['at_I'] (mm, frame I) with its normal down, turned a['twist'] about the vertical, as a press
        would hold it at contact. From the reference, ref_up_mm above the pose along z_I, the tool goes straight down to
        the pose and back: the outer part at v_mm_s, the last slow_mm at v_slow_mm_s (actual speeds at the plan's speed
        slider), dwelling dwell_ref_s at the reference before and after and dwell_s at the pose. A change of the HEX-E's
        force from the first dwell's tare, the length of its last 10 samples' mean over contact_N (default 0.5 N: the
        HEX-E's z noise is 0.24 N at one sigma, 0.08 N over 10 samples, and the simulator's reached 0.30 N), in a move
        or a dwell, is a contact: abort. Near a held tip the magnets pull or push on the robot's tip and the HEX-E reads
        that force too: an action near one sets contact_N above it. The record gives the dwells' windows (t_ref0,
        t_pose, t_ref1), the moves' (t_down, t_up) and peak_N, the largest such change over the action."""
        tg = a["target"]
        R = self.geo.R_IP(tg["n"], a.get("twist", 0.0))
        at = np.asarray(a["at_I"], float)
        T_pose = self.T_BE_for(Tm(R, at - R @ np.asarray(tg["p"], float)))
        up_B = self.reg.T_BI[:3, :3] @ EZ
        h = float(a.get("ref_up_mm", 40.0))
        slow = min(float(a.get("slow_mm", 5.0)), h)
        T_ref = Tm(T_pose[:3, :3], T_pose[:3, 3] + up_B * h)
        T_slow = Tm(T_pose[:3, :3], T_pose[:3, 3] + up_B * slow)
        for T in (T_ref, T_pose):
            if not self.in_box(T):
                raise Abort("hover pose outside the box around the indenter")
        k = float(self.plan.get("speed_slider") or 1.0)
        v_fast, v_slow = float(a.get("v_mm_s", 10.0)) / k * 1e-3, float(a.get("v_slow_mm_s", 2.0)) / k * 1e-3
        lim = float(a.get("contact_N", 0.5))
        self.travel(T_ref)
        rec = dict(id=a["id"], T_ref=T_ref, T_pose=T_pose, contact_N=lim)
        tare, _, _ = self.mean_ft(0.5)
        rec["tare"] = tare
        peak = [0.0]

        def watch(T=None, dur=0.0, v=None):
            """Hold for dur s (T None), or move to T at v m/s; either way the HEX-E is watched."""
            hist, t0 = deque(maxlen=10), self.now()
            if T is not None:
                self.be.robot.moveL(T, v, 0.1, asynchronous=True)
            while True:
                self.sleep(0.01)
                self.check_ft(tare)
                _, v = self.be.ft.ring.latest()
                hist.append(v[:3] - tare[:3])
                f = float(np.linalg.norm(np.mean(hist, axis=0)))     # the mean first: |noise| alone averages 0.26 N
                if len(hist) < hist.maxlen:
                    continue
                peak[0] = max(peak[0], f)
                if f > lim:
                    self.be.robot.stopL(3.0)
                    raise Abort(f"contact: {f:.2f} N on the HEX-E (contact_N {lim:g})")
                if T is None:
                    if self.now() - t0 >= dur:
                        return [t0, self.now()]
                elif self.be.robot.steady() and self.now() - t0 > 0.3:
                    e = pose_error(self.be.robot.pose(), T)
                    if e[0] > 0.5 or e[1] > 0.2:
                        raise Abort(f"the robot stopped {e[0]:.2f} mm, {e[1]:.2f} deg from its target")
                    return [t0, self.now()]

        rec["t_ref0"] = watch(dur=float(a.get("dwell_ref_s", 1.5)))
        rec["t_down"] = [watch(T_slow, v=v_fast)[0], watch(T_pose, v=v_slow)[1]]
        rec["t_pose"] = watch(dur=float(a.get("dwell_s", 2.0)))
        rec["t_up"] = [watch(T_slow, v=v_slow)[0], watch(T_ref, v=v_fast)[1]]
        rec["t_ref1"] = watch(dur=float(a.get("dwell_ref_s", 1.5)))
        rec["peak_N"] = peak[0]
        return rec

    # ---------------------------------------------------------------- the whole
    def speed_slider(self):
        """The plan's speed_slider: set on the controller and read back before any motion. Every speed of the program and
        the plan is scaled by it; sessions before 2026-10-25 ran at 0.30 without knowing it."""
        f = self.plan.get("speed_slider")
        if f is None:
            return
        f = float(f)
        try:
            self.be.robot.set_speed_fraction(f)
        except Exception as e:                                   # noqa: BLE001  (a controller without the input)
            print(f"WARNING the speed slider could not be set ({e}); checking its value", flush=True)
        self.sleep(0.3)
        got = self.be.robot.speed_fraction()
        self.log.event(event="speed_slider", id="run", rec=dict(set=f, read=got))
        print(f"speed slider {got:.2f} (the plan's {f:.2f})", flush=True)
        if abs(got - f) > 0.02:
            raise SystemExit(f"the speed slider reads {got:.2f} and the plan asks {f:.2f}: stopping before any motion")

    def check_slider(self):
        """The plan's speed slider read again before each action: a change at the pendant scales every speed of the run
        unseen (11-03: 0.30 to about 0.50 at 15:42; s2sph2's 3 N presses then stopped 0.45 mm deeper, peaking at 5 to
        7.8 N, for 30 min)."""
        f = self.plan.get("speed_slider")
        if f is None:
            return
        got = self.be.robot.speed_fraction()
        if abs(got - float(f)) > 0.02:
            self.log.event(event="speed_slider", id="run", rec=dict(plan=float(f), read=got))
            raise SystemExit(f"the speed slider reads {got:.2f} and the plan asks {float(f):.2f}: stopping before the "
                             "next action")

    def run(self, stages):
        if ("wall" in stages and not self.dry and "register_probe" not in self.reg.notes
                and "register" not in stages[:stages.index("wall")]):
            raise SystemExit("the walls aim with the pad's fit, and the low touch has 0.35 mm of face: "
                             "--stages ...,pad,register,wall,register,...")
        self.speed_slider()
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
            elif s == "wall":
                if not self.dry and "register_probe" not in self.reg.notes:
                    raise SystemExit("the walls aim with the pad's fit, and register made none")
                self.stage_touch("wall", "wall")
            elif s == "register":
                self.stage_register()
            elif s == "pivot":
                self.stage_pivot()
            elif s in ("grid", "shear", "roll", "twist", "rod"):
                self.stage_presses(s)
            elif s == "final":
                self.stage_presses("final")
                self.stage_gravity("final_gravity")
            elif s == "hover":
                self.stage_hover()
            elif s == "return":
                self.stage_return()
            else:
                raise SystemExit(f"unknown stage {s}")
        self.log.save("registration.json", self.reg.to_json())


STAGES = ("preflight", "homing", "gravity", "mount", "pad", "register", "pivot", "grid", "shear", "roll", "twist", "hover",
          "final")


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
    ap.add_argument("--held-board", default=None, help="serial port of the held tips' boards (firmware/"
                    "reskin5x_mux_stream behind an I2C mux), recorded into held.bin; give --board too when it is set")
    ap.add_argument("--home-from", default=None, help="FILE:LABEL, a recorded seating on the key (JSON lines with label "
                    "and T_tcp_mm) in place of the three freedrive seatings: for runs with nobody at the robot")
    ap.add_argument("--frame", default=None, help="JSON with T_BA: the fixture's frame in B measured otherwise (lab "
                    "20261102-strip_frame_two_keys.json, both keys of the strip), in place of the homing's own; the homing "
                    "still gives the seating, its joints and the line out of the key")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--wrist-window", default=None, help="deg: joint 5 stays within LO,HI in every move and target; "
                    "default the plan's safety.wrist_window, else 30,100 (the lab plan of 2026-10-22 needs 37 to 90 deg "
                    "outside the mount stage). The lift and the home path of a probe plan hold to "
                    "safety.travel_wrist_window instead")
    ap.add_argument("--gravity-expect-N", type=float, default=None, help="N: the weight of an earlier sweep with the same "
                    "tool; a fitted weight over 5 %% from it stops the session before any contact (2026-10-22: 1.567 N "
                    "in first2, before the HEX-E's cable port was squeezed)")
    ap.add_argument("--gravity-max-rms", type=float, default=0.1, help="N: the gravity sweep's force residual past which "
                    "the session stops before any contact (Table 2 expects under 0.02 N)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dry", type=float, default=0.0, help="mm added to every standoff, in probe mode back along the "
                    "stem: no contact")
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
        be = RealBackend(a, log, probe=plan["geometry"].get("probe") is not None)
    log.save("plan.json", plan) if not a.resume else None
    log.save("session.json", dict(args=vars(a), started=time.strftime("%Y-%m-%d %H:%M:%S"), stages=a.stages,
                                  ft_columns="t_host seq status Fx Fy Fz Tx Ty Tz (N, N m)",
                                  robot_columns="t_host t_ctrl q0..q5 (rad) x y z (m) rx ry rz safety_mode",
                                  board_dtype="t <f8, r <f4 x 20 (5 chips x [T C, x, y, z uT])",
                                  held_dtype="t <f8, b <u1 (mux channel), r <f4 x 20 (5 chips x [T C, x, y, z uT])"))
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
