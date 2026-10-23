#!/usr/bin/env python3
"""Simulated robot and Net F/T box for scripts/ati_calibration_session.py --sim: the session runs unchanged in virtual
time and writes the same files as on the lab's UR5, plus sim_truth.json, so that scripts/ati_calibration_analysis.py
can be scored against a known calibration.

The truth: a Mini45 whose gage values are g = D w + g0 + drift(t) + noise, D the inverse of a 6 x 6 calibration with
the row shapes of ATI's FT31565 (the owner's handoff note, R_31565) at the lab sensor's US-120-160 range (about 0.08 N
per count), gage gains up to 10 % off; the ATI turned 1.5 deg about its axis and 0.3 mm off its nominal place in the
stack; the hook cross's eyes 0.2 mm off their drawing and their holes 0.1 mm smaller than drawn (or than --eye-hole-mm),
the line (--line-mm across) knotted behind each eye's wall and leaving it as the analysis's line_point has it; the tool
beyond the ATI 120 g at (0.5, -0.3, 35) mm; a drift of up to 2 counts per minute per
gage and a random walk; 3 counts of noise per sample at 500 Hz; the robot's flange 0.03 deg off its kinematic model at
each stop; a mass that swings after each move with 0.3 to 1.5 deg, decaying in 6 s, and 3 deg after it is hung.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT / "src/morphohand/driver/reskin5x/host"
sys.path.insert(0, str(HOST))
sys.path.insert(0, str(ROOT / "scripts"))
import ati_calibration_analysis as AN  # noqa: E402
import ur5_tip_calibration as TC  # noqa: E402
import ur5_tip_calibration_sim as URSIM  # noqa: E402

R_31565 = np.array([[-0.026289, 0.000278, 0.056518, -0.702557, -0.032851, 0.708130],
                    [-0.095998, 0.795815, 0.015191, -0.420373, 0.026813, -0.424028],
                    [0.575115, 0.028712, 0.571098, 0.027078, 0.583951, 0.023091],
                    [-0.033322, 0.232680, -0.670809, -0.155471, 0.679074, -0.097216],
                    [0.774309, 0.038323, -0.420813, 0.191089, -0.366102, -0.226602],
                    [0.079040, -0.547565, 0.009397, -0.583523, 0.043265, -0.592846]])
RATE = 500.0
ROBOT_RATE = 125.0


def rotz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def small_rot(rng, sd_deg):
    w = rng.normal(0.0, math.radians(sd_deg), 3)
    return TC.exp_so3(w)


class Truth:
    def __init__(self, plan, seed=0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        # counts -> N, N m: FT31565's row shapes; row norms such that the US-120-160 range spans about 20000 counts
        scale = np.array([0.11, 0.11, 0.11, 0.0037, 0.0037, 0.0037])
        self.gains = 1.0 + rng.uniform(-0.1, 0.1, 6)
        self.C = np.diag(scale) @ R_31565 @ np.diag(self.gains)
        self.D = np.linalg.inv(self.C)
        self.g0 = rng.uniform(-3000, 3000, 6)
        self.drift = rng.uniform(-2.0, 2.0, 6) / 60.0                 # counts per s
        self.walk = 0.02                                              # counts per sqrt(s)
        self.noise = 3.0
        self.clock_err, self.off_err = 1.5, rng.normal(0.0, 0.3, 3)
        geo = plan["geometry"]
        self.T_EA_nom = np.array(geo["T_EA"], float)
        R_err = rotz(self.clock_err)
        self.T_EA = self.T_EA_nom.copy()
        self.T_EA[:3, :3] = self.T_EA_nom[:3, :3] @ R_err
        self.T_EA[:3, 3] = self.T_EA_nom[:3, 3] + self.T_EA_nom[:3, :3] @ self.off_err
        # the eyes in the true ATI frame: the nominal ATI-frame points, moved by the printing error, seen from the turned,
        # shifted ATI
        self.eyes = {}
        for e in geo["eyes"]:
            p_nom = np.array(e["p_A"], float) + rng.normal(0.0, 0.2, 3)
            p = R_err.T @ (p_nom - self.off_err)
            self.eyes[e["name"]] = (p, R_err.T @ np.array(e["u_A"], float))
        self.hole_r = plan["geometry"]["eye"]["hole_d"] / 2 - 0.05            # the holes print 0.1 mm small
        self.wall = plan["geometry"]["eye"].get("wall", 1.6)
        self.line_d = 0.0
        self.m_tool, self.c_tool = 0.120, np.array([0.5, -0.3, 35.0])
        self.mass_kg = None
        self.g = plan["g_local"]

    def to_json(self):
        return dict(C=self.C, D=self.D, gains=self.gains, g0=self.g0, drift_counts_per_s=self.drift, noise=self.noise,
                    clock_err_deg=self.clock_err, origin_err_mm=self.off_err, hole_r_mm=self.hole_r, wall_mm=self.wall,
                    line_d_mm=self.line_d, m_tool=self.m_tool,
                    c_tool_mm=self.c_tool, mass_kg=self.mass_kg, eyes={k: dict(p_A=p, u_A=u) for k, (p, u) in self.eyes.items()},
                    T_EA_true=self.T_EA)


class World:
    def __init__(self, plan, truth: Truth, q0):
        self.plan, self.tr = plan, truth
        self.t = 0.0
        self.q = np.array(q0, float)
        self.traj = None                                              # (t0, q0, q1, T, v, a)
        self.mass = None                                              # the eye the mass hangs on
        self.swing = None                                             # (t0, amplitude rad, direction (base), period)
        self.R_err = np.eye(3)
        self.walk = np.zeros(6)
        self.ati_app = self.robot_app = None
        self.ati_ring, self.robot_ring = TC.Ring(keep_s=120.0), TC.Ring(keep_s=120.0)
        self.seq = 0
        self.n = 0

    # robot -----------------------------------------------------------------------------------------------------------
    def q_at(self, t):
        if self.traj is None:
            return self.q
        t0, qa, qb, T, v, a = self.traj
        s = min(max(t - t0, 0.0), T)
        d = float(np.abs(qb - qa).max())
        if d < 1e-12:
            return qb
        ta = v / a
        if T <= 2 * ta:                                               # triangle
            h = T / 2
            x = 0.5 * a * s * s if s <= h else d - 0.5 * a * (T - s) ** 2
        else:
            x = 0.5 * a * s * s if s <= ta else (v * (s - ta / 2) if s <= T - ta else d - 0.5 * a * (T - s) ** 2)
        return qa + (qb - qa) * min(x / d, 1.0)

    def move(self, q1, v, a):
        q0 = self.q_at(self.t)
        d = float(np.abs(q1 - q0).max())
        T = d / v + v / a if d > v * v / a else 2 * math.sqrt(max(d, 1e-12) / a)
        self.traj = (self.t, q0, np.array(q1, float), T, v, a)

    def moving(self):
        return self.traj is not None and self.t < self.traj[0] + self.traj[3]

    # the box -----------------------------------------------------------------------------------------------------------
    def wrench(self, q, t):
        tr = self.tr
        T_BA = URSIM.fk(q) @ tr.T_EA
        R_BA = T_BA[:3, :3] @ self.R_err
        g_B = np.array([0.0, 0.0, -tr.g])
        g_A = R_BA.T @ g_B
        w = np.r_[tr.m_tool * g_A, np.cross(tr.c_tool * 1e-3, tr.m_tool * g_A)]
        if self.mass is not None:
            p, u = tr.eyes[self.mass]
            f = tr.mass_kg * g_A
            if self.swing is not None:
                t0, amp, dirB, per = self.swing
                th = amp * math.exp(-(t - t0) / 6.0) * math.sin(2 * math.pi * (t - t0) / per)
                f = f + tr.mass_kg * tr.g * th * (R_BA.T @ dirB)
            pe = AN.line_point(p, u, f / np.linalg.norm(f), tr.hole_r, tr.wall, tr.line_d)
            w = w + np.r_[f, np.cross(pe * 1e-3, f)]
        return w

    def advance(self, dt):
        t1 = self.t + dt
        ts = np.arange(math.ceil(self.t * ROBOT_RATE), math.floor(t1 * ROBOT_RATE) + 1) / ROBOT_RATE
        ts = ts[(ts >= self.t) & (ts < t1)]
        was_moving = self.moving()
        rows, W = [], []
        for t in ts:
            q = self.q_at(t)
            T = URSIM.fk(q)
            pose = TC.T_to_ur(T)
            row = [t, t, *q, *pose, 1.0]
            rows.append(row)
            self.robot_ring.add(t, np.array(row[1:]))
            W.append(self.wrench(q, t))
        if self.robot_app is not None and rows:
            self.robot_app.write(rows)
        tg = np.arange(math.ceil(self.t * RATE), math.floor(t1 * RATE) + 1) / RATE
        tg = tg[(tg >= self.t) & (tg < t1)]
        if len(tg) and len(W):
            W = np.array(W)
            Wi = np.column_stack([np.interp(tg, ts, W[:, k]) for k in range(6)]) if len(ts) > 1 else np.repeat(W, len(tg), 0)
            tr = self.tr
            steps = tr.rng.normal(0.0, tr.walk * math.sqrt(1.0 / RATE), (len(tg), 6))
            walk = self.walk + np.cumsum(steps, axis=0)
            self.walk = walk[-1]
            G = Wi @ tr.D.T + tr.g0 + np.outer(tg, tr.drift) + walk + tr.rng.normal(0.0, tr.noise, (len(tg), 6))
            G = np.round(G)
            out = []
            for t, g in zip(tg, G):
                self.seq += 1
                self.ati_ring.add(t, g)
                out.append((t, self.seq, self.seq * 14, 0, *g))
            self.n += len(out)
            if self.ati_app is not None:
                self.ati_app.write(out)
        self.t = t1
        if was_moving and not self.moving():                          # a stop: the flange's own error, the mass's swing
            self.q = self.traj[2]
            self.R_err = small_rot(self.tr.rng, 0.03)
            if self.mass is not None:
                a = self.tr.rng.uniform(0.0, 2 * math.pi)
                L = (self.plan["mass"]["line_mm"] + self.plan["mass"]["h_mm"] / 2) * 1e-3
                self.swing = (self.t, math.radians(self.tr.rng.uniform(0.3, 1.5)), np.array([math.cos(a), math.sin(a), 0.0]),
                              2 * math.pi * math.sqrt(L / self.tr.g))


class SimRobot:
    def __init__(self, w: World):
        self.w = w
        self.ring = w.robot_ring

    def q(self):
        return self.w.q_at(self.w.t)

    def pose(self):
        return URSIM.fk(self.q())

    def moveJ(self, q, v=0.3, a=0.3, asynchronous=False):
        self.w.move(np.array(q, float), v, a)

    def steady(self):
        return not self.w.moving()

    def stopped(self):
        return False

    def stopL(self, a=0.5):
        self.w.traj = None

    stopJ = stopL

    def set_payload(self, kg, cog_mm):
        pass

    def close(self):
        pass


class SimAti:
    def __init__(self, w: World):
        self.w = w
        self.ring = w.ati_ring
        self.lost, self.gaps, self.bad_status, self.stop = 0, [], [], False

    @property
    def n(self):
        return self.w.n


class SimBackend:
    def __init__(self, plan, args, seed=0):
        self.truth = Truth(plan, seed)
        self.truth.mass_kg = (args.mass_g or 500.0) * 1e-3
        if getattr(args, "eye_hole_mm", None):
            self.truth.hole_r = args.eye_hole_mm / 2 - 0.05
        self.truth.line_d = getattr(args, "sim_line_mm", None) or getattr(args, "line_mm", None) or 0.0
        self.w = World(plan, self.truth, np.radians(plan["home_q_deg"]))
        self.robot, self.ati, self.hexe = SimRobot(self.w), SimAti(self.w), None
        self.log = None

    def attach(self, log):
        self.log = log
        self.w.ati_app = log.appender("ati.bin")
        self.w.robot_app = log.appender("robot.bin")
        log.save("sim_truth.json", self.truth.to_json())

    def config(self):
        cfg = dict(cfgcalsn="FT00000", cfgcpf="1", cfgcpt="1", comrdte="1", setcfgsel="15", scfgfu="lbf", scfgtu="lbf-in",
                   comrdtrate="500", setuserfilter="0")
        return "<netft simulated/>", cfg

    def now(self):
        return self.w.t

    def sleep(self, dt):
        while dt > 1e-9:
            h = min(dt, 0.5)
            self.w.advance(h)
            dt -= h

    def operator(self, prompt):
        self.sleep(20.0)                                              # the operator at the robot
        if prompt.startswith("hang the mass on the "):
            self.w.mass = prompt.split("hang the mass on the ")[1].split(" eye")[0]
            a = self.truth.rng.uniform(0.0, 2 * math.pi)
            L = (self.w.plan["mass"]["line_mm"] + self.w.plan["mass"]["h_mm"] / 2) * 1e-3
            self.w.swing = (self.w.t, math.radians(3.0), np.array([math.cos(a), math.sin(a), 0.0]),
                            2 * math.pi * math.sqrt(L / self.truth.g))
        elif prompt.startswith("take the mass off"):
            self.w.mass, self.w.swing = None, None
        return ""

    def close(self):
        pass
