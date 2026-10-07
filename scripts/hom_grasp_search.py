#!/usr/bin/env python3
"""Grasp search by turn range for the fixed-contact three-finger turn (prototype, CPU; 2026-10-06).

The HOM controller of scripts/hom_turn3.py keeps all three contacts fixed on the tool and stops at 6-42 deg on the
deployed grasps, mostly at a pip limit. This searches the grasp for the turn instead.

Turn range of a grasp: the largest rotation theta of the tool toward vertical, about the horizontal axis normal to its
starting axis through the centroid of the three contacts (the rotation hom_turn3 commands), such that for every
theta' <= theta (3 deg steps)
  * each fingertip reaches its tool-fixed contact point: damped least squares IK of the finger's three joints puts
    the fingertip's pad point (the deployed grip's witness point on the TPU block, in the tip frame) within 1 mm of
    the contact, inside the joint limits less 3 deg (the governor's margin), with the pad normal within 45 deg of the
    tool's inward surface normal;
  * pad forces exist that hold the tool: an LP over the three contact forces with Sum f = the tool's weight,
    Sum (c - com) x f = 0, each force in the mu = 1 friction pyramid (8 facets, inscribed) with at least 0.5 N
    normal, and every finger joint torque J^T f within the 1 N m servo limit.
Grasp: the tool's offset on its post (axial a, lateral b, yaw psi) and each finger's contact on the 25 mm cylinder
(axial position s, angle phi around the axis). CEM (48 samples, 8 elites, 12 iterations) maximises the turn range,
starting from the deployed grip's contacts (mj_geomDistance witness points of the plan grip). Run: `search` writes one
row per hand; `hom` runs hom_turn3's controller from the best grasp (grip targets = the IK solution at theta 0 with
the pad commanded 2 mm into the tool, fingers starting from a pre-grasp with the pads 8 mm outside their contacts)
and from the deployed grasp (the plan's open pose and grip), in MuJoCo (1 mm pads) and Drake. --lim-deg widens the
search's joint-limit margin; at 3 deg a searched start pose can sit exactly where hom_turn3's governor stops.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/hom_grasp_search.py search --hands D7 D2 D5
    $PY scripts/hom_grasp_search.py hom --hands D7 D2 D5 --sims mujoco drake
Rows: docs/experiments/20261006-hom_turn3/grasp_search.jsonl, turn3_best_grasp.jsonl.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_turn3 as H3  # noqa: E402

RB = H3.RB
FINGERS, JOINTS, NAMES = H3.FINGERS, H3.JOINTS, H3.NAMES
D = ROOT / "docs/experiments/20261006-hom_turn3"
OUT_SEARCH = D / "grasp_search.jsonl"
OUT_HOM = D / "turn3_best_grasp.jsonl"
MU, F_MIN, TAU_MAX, G = 1.0, 0.5, 1.0, 9.81
STEP_DEG, GOAL_DEG = 3.0, 90.0
LIM_DEG = 3.0
IK_TOL, NORMAL_DEG = 0.001, 45.0
SQUEEZE, PRE = 0.002, 0.008
BOUNDS = dict(a=0.012, b=0.005, psi=math.radians(15.0))
OPT = {"lim_deg": LIM_DEG}       # joint-limit margin of the search; hom_turn3's governor stops within 3 deg of a limit


def quat_mul(q, r):
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1, w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1, w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])


def quat_axis(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis]


def rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Hand:
    """Kinematics of one hand on its bench scene (the Mirror: TPU block as one convex mesh per tip)."""

    def __init__(self, hand):
        self.hand = hand
        p_pads, p_mesh, meta, meta_mesh = H3.scenes(hand, H3.PLANT, 1.0)
        self.paths = (p_pads, p_mesh)
        self.meta, self.meta_mesh = meta, meta_mesh
        self.mir = H3.Mirror(p_mesh)
        m = self.mir.m
        self.m_tool = float(m.body_subtreemass[self.mir.tb])
        lim = math.radians(OPT["lim_deg"])
        self.lo, self.hi = self.mir.lo + lim, self.mir.hi - lim
        tool7 = np.asarray(meta["tool7"], float)
        self.p0, self.q0 = tool7[:3], tool7[3:] / np.linalg.norm(tool7[3:])
        R0 = rot(self.q0)
        self.u0 = R0[:, 2]
        self.lat = np.cross(self.u0, [0.0, 0.0, 1.0])
        self.lat /= np.linalg.norm(self.lat)
        plan, _ = RB.load_plan(hand)
        poses = {p["name"]: p["joints"] for p in plan["poses"]}
        self.grip = np.array([math.radians(poses["grip"][f][j]) for f in FINGERS for j in JOINTS])
        # deployed contacts: witness points of the plan grip on the deployed tool pose
        self.mir.set_state(self.grip, self.p0, self.q0)
        self.pad_pt, self.pad_n, x0 = {}, {}, []
        for f in FINGERS:
            g = self.mir.gcf(f)
            tb = self.mir.tipb[f]
            Rt, pt = self.mir.d.xmat[tb].reshape(3, 3), self.mir.d.xpos[tb]
            self.pad_pt[f] = Rt.T @ (g["p_f"] - pt)
            self.pad_n[f] = Rt.T @ (-g["n"])            # pad's outward normal, toward the tool
            loc = R0.T @ (g["p_o"] - self.p0)
            x0 += [float(loc[2]), math.atan2(loc[1], loc[0])]
        self.x_deployed = np.r_[0.0, 0.0, 0.0, x0]
        self.r = self.mir.r_cyl
        self.hl = self.mir.hl

    def tool_pose(self, x):
        a, b, psi = x[:3]
        p = self.p0 + a * self.u0 + b * self.lat
        q = quat_mul(quat_axis([0, 0, 1], psi), self.q0)
        return p, q / np.linalg.norm(q)

    def contacts_local(self, x):
        out = {}
        for k, f in enumerate(FINGERS):
            s, phi = x[3 + 2 * k], x[4 + 2 * k]
            out[f] = (np.array([self.r * math.cos(phi), self.r * math.sin(phi), s]),
                      np.array([math.cos(phi), math.sin(phi), 0.0]))
        return out

    def pad_world(self, f):
        d = self.mir.d
        tb = self.mir.tipb[f]
        Rt = d.xmat[tb].reshape(3, 3)
        return d.xpos[tb] + Rt @ self.pad_pt[f], Rt @ self.pad_n[f]

    def ik(self, f, q, target, iters=40, lam=0.01):
        """Finger f's three joints (others fixed in q) so that its pad point reaches target; returns q, residual."""
        k = FINGERS.index(f)
        sl = slice(3 * k, 3 * k + 3)
        q = q.copy()
        mir = self.mir
        err = None
        for _ in range(iters):
            mir.d.qpos[mir.qadr] = q
            mir.mj.mj_kinematics(mir.m, mir.d)
            mir.mj.mj_comPos(mir.m, mir.d)
            p, _ = self.pad_world(f)
            e = target - p
            err = float(np.linalg.norm(e))
            if err < 2e-4:
                break
            J, _ = mir.jac(f, p)
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(3), e)
            q[sl] = np.clip(q[sl] + dq, self.lo[sl], self.hi[sl])
        mir.d.qpos[mir.qadr] = q
        mir.mj.mj_kinematics(mir.m, mir.d)
        mir.mj.mj_comPos(mir.m, mir.d)
        p, _ = self.pad_world(f)
        return q, float(np.linalg.norm(target - p))

    def forces_ok(self, C, N, com):
        """LP: pad forces holding the tool (weight, moments about its centre of mass) within cones and torques."""
        n_f = len(FINGERS)
        A_eq = np.zeros((6, 3 * n_f))
        b_eq = np.r_[0.0, 0.0, self.m_tool * G, 0.0, 0.0, 0.0]
        A_ub, b_ub = [], []
        c = np.zeros(3 * n_f)
        for i, f in enumerate(FINGERS):
            sl = slice(3 * i, 3 * i + 3)
            A_eq[0:3, sl] = np.eye(3)
            rx = C[f] - com
            A_eq[3:6, sl] = np.array([[0, -rx[2], rx[1]], [rx[2], 0, -rx[0]], [-rx[1], rx[0], 0]])
            n = N[f]                                     # outward tool normal; the finger pushes along -n
            t1 = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(n, t1)
            row = np.zeros(3 * n_f)
            row[sl] = n                                  # n.f <= -F_MIN
            A_ub.append(row)
            b_ub.append(-F_MIN)
            mu_in = MU * math.cos(math.pi / 8)
            for kk in range(8):
                t = math.cos(kk * math.pi / 4) * t1 + math.sin(kk * math.pi / 4) * t2
                row = np.zeros(3 * n_f)
                row[sl] = t + mu_in * n                  # t.f <= mu (-n.f)
                A_ub.append(row)
                b_ub.append(0.0)
            J, _ = self.mir.jac(f, C[f])
            for jrow in J.T:                             # |J^T f| <= TAU_MAX (the finger feels -f)
                row = np.zeros(3 * n_f)
                row[sl] = jrow
                A_ub.append(row.copy())
                b_ub.append(TAU_MAX)
                A_ub.append(-row)
                b_ub.append(TAU_MAX)
            c[sl] = -n                                   # minimise the total normal force
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=b_eq,
                      bounds=[(None, None)] * (3 * n_f), method="highs")
        return res.status == 0, (res.x if res.status == 0 else None)

    def turn_range(self, x, detail=False):
        """Largest feasible theta (deg) of grasp x; -STEP_DEG if theta 0 fails. With detail, the per-step record."""
        p, q = self.tool_pose(x)
        R = rot(q)
        loc = self.contacts_local(x)
        C0 = {f: p + R @ c for f, (c, n) in loc.items()}
        pivot = np.mean([C0[f] for f in FINGERS], axis=0)
        u = R[:, 2]
        a0 = np.cross(u, [0.0, 0.0, 1.0])
        a0 /= np.linalg.norm(a0)
        qj = self.grip.copy()
        best, rec, why = -STEP_DEG, [], None
        for th_deg in np.arange(0.0, GOAL_DEG + 1e-9, STEP_DEG):
            Rq = rot(quat_axis(a0, math.radians(th_deg)))
            pt, Rt = pivot + Rq @ (p - pivot), Rq @ R
            self.mir.set_state(qj, pt, quat_mul(quat_axis(a0, math.radians(th_deg)), q))
            com = self.mir.d.xipos[self.mir.tb].copy()
            C = {f: pt + Rt @ c for f, (c, n) in loc.items()}
            N = {f: Rt @ n for f, (c, n) in loc.items()}
            ok = True
            for f in FINGERS:
                qj, res = self.ik(f, qj, C[f])
                _, pn = self.pad_world(f)
                ang = math.degrees(math.acos(max(-1.0, min(1.0, float(pn @ -N[f])))))
                if res > IK_TOL:
                    ok, why = False, f"{f} reach {1e3 * res:.1f} mm"
                    break
                if ang > NORMAL_DEG:
                    ok, why = False, f"{f} pad normal {ang:.0f} deg"
                    break
            if ok:
                fo, fx = self.forces_ok(C, N, com)
                if not fo:
                    ok, why = False, "forces"
            if detail:
                rec.append({"th_deg": float(th_deg), "ok": ok, "q_deg": np.degrees(qj).round(2).tolist(),
                            "why": None if ok else why})
            if not ok:
                break
            best = float(th_deg)
        return (best, why, rec) if detail else best

    def clamp(self, x):
        x = x.copy()
        x[0] = np.clip(x[0], -BOUNDS["a"], BOUNDS["a"])
        x[1] = np.clip(x[1], -BOUNDS["b"], BOUNDS["b"])
        x[2] = np.clip(x[2], -BOUNDS["psi"], BOUNDS["psi"])
        for k in range(3):
            x[3 + 2 * k] = np.clip(x[3 + 2 * k], -(self.hl - 0.008), self.hl - 0.008)
        return x

    def pregrasp(self, x, clearance=PRE):
        """IK at theta 0 with each pad `clearance` outside its contact: the fingers' starting pose for a rollout."""
        p, q = self.tool_pose(x)
        R = rot(q)
        self.mir.set_state(self.grip, p, q)
        qj = self.grip.copy()
        for f, (c, n) in self.contacts_local(x).items():
            qj, _ = self.ik(f, qj, p + R @ (c + clearance * n))
        return qj

    def grip_targets(self, x):
        """IK at theta 0 with each pad commanded SQUEEZE into the tool: the grip pose for a rollout."""
        p, q = self.tool_pose(x)
        R = rot(q)
        self.mir.set_state(self.grip, p, q)
        qj = self.grip.copy()
        for f, (c, n) in self.contacts_local(x).items():
            qj, _ = self.ik(f, qj, p + R @ (c - SQUEEZE * n))
        return qj


def cem(hand, iters=12, pop=48, elites=8, seed=0):
    rng = np.random.default_rng(seed)
    mu = hand.x_deployed.copy()
    sd = np.r_[0.006, 0.002, math.radians(6.0), [0.010, math.radians(15.0)] * 3]
    best_x, best_s = mu.copy(), hand.turn_range(mu)
    hist = []
    for it in range(iters):
        X = np.array([hand.clamp(mu + sd * rng.standard_normal(len(mu))) for _ in range(pop)])
        X[0] = hand.clamp(best_x)
        S = np.array([hand.turn_range(x) for x in X])
        order = np.argsort(-S)
        E = X[order[:elites]]
        mu = E.mean(axis=0)
        sd = np.maximum(E.std(axis=0), 0.25 * np.r_[0.006, 0.002, math.radians(6.0), [0.010, math.radians(15.0)] * 3])
        if S[order[0]] > best_s:
            best_s, best_x = float(S[order[0]]), X[order[0]].copy()
        hist.append({"iter": it, "best": best_s, "elite_mean": float(S[order[:elites]].mean()),
                     "pop_mean": float(S.mean())})
        print(f"  {hand.hand} iter {it}: best {best_s:.0f} deg, elite mean {hist[-1]['elite_mean']:.1f}", flush=True)
    return best_x, best_s, hist


def describe(hand, x):
    rng, why, rec = hand.turn_range(x, detail=True)
    return {"range_deg": rng, "stop": why, "a_mm": 1e3 * x[0], "b_mm": 1e3 * x[1], "psi_deg": math.degrees(x[2]),
            "contacts": {f: {"s_mm": 1e3 * x[3 + 2 * k], "phi_deg": math.degrees(x[4 + 2 * k])}
                         for k, f in enumerate(FINGERS)},
            "q_start_deg": rec[0]["q_deg"] if rec else None, "q_end_deg": rec[-1]["q_deg"] if rec else None,
            "x": [float(v) for v in x]}


def append(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def cmd_search(a):
    for hd in a.hands:
        w0 = time.perf_counter()
        hand = Hand(hd)
        dep = describe(hand, hand.x_deployed)
        print(f"{hd}: deployed grasp range {dep['range_deg']:.0f} deg, stop: {dep['stop']}", flush=True)
        bx, bs, hist = cem(hand, iters=a.iters, pop=a.pop)
        best = describe(hand, bx)
        append(OUT_SEARCH, {"hand": hd, "deployed": dep, "best": best, "cem": hist, "iters": a.iters, "pop": a.pop,
                            "wall_s": round(time.perf_counter() - w0, 1), "when": time.strftime("%Y-%m-%d %H:%M"),
                            "criteria": {"step_deg": STEP_DEG, "lim_deg": OPT["lim_deg"], "ik_tol_mm": 1e3 * IK_TOL,
                                         "normal_deg": NORMAL_DEG, "mu": MU, "f_min_N": F_MIN, "tau_max_Nm": TAU_MAX}})
        print(f"{hd}: best grasp range {best['range_deg']:.0f} deg (stop: {best['stop']}), "
              f"{time.perf_counter() - w0:.0f} s", flush=True)


def cmd_hom(a):
    rows = {}
    for line in open(OUT_SEARCH):
        r = json.loads(line)
        if abs(r["criteria"]["lim_deg"] - OPT["lim_deg"]) < 1e-9:
            rows[r["hand"]] = r
    prm = dict(H3.DEFAULTS)
    load_plan = RB.load_plan
    for hd in a.hands:
        hand = Hand(hd)
        for which in a.grasps:
            x = np.asarray(rows[hd][which]["x"])
            meta, meta_mesh = copy.deepcopy(hand.meta), copy.deepcopy(hand.meta_mesh)
            plan, traj = load_plan(hd)
            if which == "best":
                p, q = hand.tool_pose(x)
                for mm in (meta, meta_mesh):
                    mm["tool7"] = [float(v) for v in np.r_[p, q]]
                qg = hand.grip_targets(x)
                q_pre = hand.pregrasp(x)          # start outside the tool: closing from the open pose knocks it off
                for mm in (meta, meta_mesh):
                    mm["q0"] = {n: float(v) for n, v in zip(NAMES, q_pre)}
                plan = copy.deepcopy(plan)
                for pose in plan["poses"]:
                    if pose["name"] == "grip":
                        for k, f in enumerate(FINGERS):
                            for j, jn in enumerate(JOINTS):
                                pose["joints"][f][jn] = math.degrees(float(qg[3 * k + j]))
            RB.load_plan = lambda h, _p=plan, _t=traj: (_p, _t)
            try:
                for sim in a.sims:
                    plant = H3.MjPlant(hand.paths[0], meta) if sim == "mujoco" else H3.DrakePlant(hand.paths[1],
                                                                                                 meta_mesh)
                    for seed in a.seeds:
                        row = {"hand": hd, "grasp": which, "sim": sim, "seed": seed, "plant": H3.PLANT,
                               "range_deg_kinematic": rows[hd][which]["range_deg"], "search_lim_deg": OPT["lim_deg"],
                               "start": "pregrasp 8 mm" if which == "best" else "plan open pose",
                               "when": time.strftime("%Y-%m-%d %H:%M")}
                        try:
                            res, _ = H3.rollout(plant, hand.mir, meta, seed, prm)
                            row.update(res, status="ok")
                        except Exception as e:
                            row.update(status="error", error=f"{type(e).__name__}: {e}",
                                       tb=traceback.format_exc()[-1200:])
                        append(OUT_HOM, row)
                        print(f"{hd} {which:<8} {sim:<6} seed {seed}: " + (
                            f"turn {row['turn_end_deg']:+.1f} deg, held {row['held']}, frozen {row['frozen_deg']} at "
                            f"{row['limit_joint']}" if row["status"] == "ok" else row["error"]), flush=True)
            finally:
                RB.load_plan = load_plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--hands", nargs="+", default=["D7", "D2", "D5"])
    s.add_argument("--iters", type=int, default=12)
    s.add_argument("--pop", type=int, default=48)
    s.add_argument("--lim-deg", type=float, default=LIM_DEG, help="joint-limit margin of the search")
    h = sub.add_parser("hom")
    h.add_argument("--hands", nargs="+", default=["D7", "D2", "D5"])
    h.add_argument("--sims", nargs="+", default=["mujoco", "drake"])
    h.add_argument("--seeds", nargs="+", type=int, default=[0])
    h.add_argument("--lim-deg", type=float, default=LIM_DEG, help="use the search rows made with this margin")
    h.add_argument("--grasps", nargs="+", default=["best", "deployed"], choices=["best", "deployed"])
    a = ap.parse_args()
    OPT["lim_deg"] = a.lim_deg
    cmd_search(a) if a.cmd == "search" else cmd_hom(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
