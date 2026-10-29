#!/usr/bin/env python3
"""Foundational-pose sampling of SR2 gantry layouts: the method of the foundational-pose paper (Wang, Oh and Pollard,
arXiv 2409.14158) run on the closed-form kinematics of the CAD-model SR2 hand, for the review page of 2026-10-29
(scripts/fp_review_page.py).

A layout is the six palm-frame mount coordinates of the three fingers (mm), inside the gantries' travel
(REAL_V1_WORKSPACE). It is a candidate when it reaches two foundational poses (FPs), the grasps of the HOM tasks
(scripts/hom_tasks.py):
  pinch   thumb and index pinch a 25 mm cylinder (100 mm long, lying flat) across its axis, and the middle finger
          touches it on the index's side at least 40 mm along the axis (the pin-push grasp: the pinch is a revolute
          joint and the middle finger turns the cylinder about it);
  tripod  three pads on the equator of a 40 mm sphere (the roll and twist grasp).
Reaching a pose is a nonlinear program over the object's pose and the contact coordinates (the paper's theta), solved
by SLSQP and warm-started from the nearest candidate's solution, as the paper's RRT does with fmincon. Its constraints
are the servo limits less 3 deg, the distal link facing the object within 25 deg and a lever of 40 mm; its objective is
the smallest singular value of the three pad Jacobians (the HOM task design's ranking score). Finger-to-finger
clearance (1 mm) and link-to-object clearance (0.5 mm) are checked on the solution.

Three metrics per pose, after the paper's tool motion range (TMR), maximum finger torque (MFT) and sliding:
  tmr_deg    rotation of the object with the contacts fixed on it, both directions summed (about the pinch axis for
             'pinch', about the vertical through the centre for 'tripod'), each direction capped at 60 deg;
  mft_mNm    the largest joint torque along that rotation under the task's grasp forces (pinch 2 N per pinching pad
             and 1 N of drive on the middle; tripod 1.5 N per pad), J^T f;
  smin_mm    the smallest pad-Jacobian singular value at the pose (mm/rad; the paper's sliding speed is zero by
             construction when the contacts are fixed).
Each metric becomes a score from 0 (worst candidate) to 100 (best), as in the paper.

  .venv/bin/python scripts/fp_sr2_toy.py run2d [--max 600]     # symmetric tripods (x_sep, y_sep), the paper's 2-D case
  .venv/bin/python scripts/fp_sr2_toy.py run6d [--max 2500]    # six mount coordinates, RRT from the seed layout
  .venv/bin/python scripts/fp_sr2_toy.py sobol [--n 1024]      # the same solver on a scrambled Sobol draw
  uv run --no-sync --with scikit-learn python scripts/fp_sr2_toy.py embed    # t-SNE and PCA of the 6-D candidates
Rows: docs/experiments/20261029-foundational_pose_review/rows/{rrt2d,rrt6d,sobol6d}.jsonl, one fsynced line per RRT
call that reached the solver; embed writes embed6d.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hand_object_scale_kin as K  # noqa: E402

OUT = ROOT / "docs/experiments/20261029-foundational_pose_review"
ROWS = OUT / "rows"
FINGERS = K.FINGERS
LIM = K.limits(False)
MARGIN = math.radians(3.0)
# the distal link must face the object: within 25 deg for the pinch (the pin-push design's gate in hom_tasks.py), in
# the pad's front half (90 deg) for the equator tripod (the scale study's gate, hand_object_scale_kin.grasp_scan)
COS_FACE = {"pinch": math.cos(math.radians(25.0)), "tripod": 0.0}
FACE_PATH = {"pinch": math.cos(math.radians(45.0)), "tripod": 0.0}     # along the TMR rotation
FF_CLEAR, OBJ_CLEAR = 0.001, 0.0005
CYL_D, CYL_L, SPH_D = 0.025, 0.100, 0.040
LEVER = 0.040
F_PINCH, F_DRIVE, F_TRIPOD = 2.0, 1.0, 1.5
TMR_CAP, TMR_STEP = 60.0, 1.0
# gantry travel, palm frame, mm: REAL_V1_MOUNTS + REAL_V1_WORKSPACE (src/morphohand/sampling/morphology.py)
LO6 = np.array([-80.0, -55.0, 20.0, 25.0, 20.0, -85.0])
HI6 = np.array([-20.0, 55.0, 80.0, 85.0, 80.0, -25.0])
NAMES6 = ("thumb_x", "thumb_y", "index_x", "index_y", "middle_x", "middle_y")
LO2, HI2 = np.array([40.0, 50.0]), np.array([160.0, 170.0])     # x_sep, y_sep of the symmetric tripod (scale study)
SEED6 = np.array([-50.0, 0.0, 50.0, 55.0, 50.0, -55.0])           # all gantries at mid-travel: the CAD-nominal hand
FPS = ("pinch", "tripod")


# ------------------------------------------------------------------------------------------------ layouts

def mounts6(d):
    """Palm-frame mounts (m) of a six-coordinate layout (mm)."""
    d = np.asarray(d, float) / 1000.0
    return {"thumb": np.array([d[0], d[1], 0.0]), "index": np.array([d[2], d[3], 0.0]),
            "middle": np.array([d[4], d[5], 0.0])}


def layout2(x_sep, y_sep):
    """The symmetric tripod of the scale study as six coordinates (mm), centred like REAL_V1_MOUNTS."""
    return np.array([-x_sep / 2, 0.0, x_sep / 2, y_sep / 2, x_sep / 2, -y_sep / 2])


# ------------------------------------------------------------------------------------------------ foundational poses

def geometry(fp, v, M=None):
    """Object spec, pad-centre targets (palm frame), outward contact normals and the task rotation axis (point, unit)
    for decision variables v. A tripod contact's azimuth is the direction from the sphere's centre to its finger's
    mount (M, the layout's mounts) plus that finger's offset in v."""
    if fp == "pinch":
        xc, yc, h, psi, sp, sm, php, phm = v
        r = CYL_D / 2
        a = np.array([-math.sin(psi), math.cos(psi), 0.0])           # cylinder axis (palm y yawed by psi)
        e = np.array([math.cos(psi), math.sin(psi), 0.0])            # across it, thumb side -e
        z = np.array([0.0, 0.0, 1.0])
        c = np.array([xc, yc, -h])
        n = {"thumb": -math.cos(php) * e - math.sin(php) * z, "index": math.cos(php) * e - math.sin(php) * z,
             "middle": math.cos(phm) * e - math.sin(phm) * z}
        s = {"thumb": sp, "index": sp, "middle": sm}
        obj = {"shape": "cylinder", "d": CYL_D, "r": r, "centre": c, "axis": a, "hl": CYL_L / 2, "length": CYL_L}
        pc = {f: c + s[f] * a + r * n[f] for f in FINGERS}              # contact points on the object
        p_axis = 0.5 * (pc["thumb"] + pc["index"])
        u_axis = pc["index"] - pc["thumb"]
        u_axis = u_axis / np.linalg.norm(u_axis)
    else:
        xc, yc, h, dt_, di, dm, lat = v
        r = SPH_D / 2
        c = np.array([xc, yc, -h])
        az = {f: math.atan2(M[f][1] - yc, M[f][0] - xc) + dd for f, dd in zip(FINGERS, (dt_, di, dm))}
        n = {f: np.array([math.cos(lat) * math.cos(az[f]), math.cos(lat) * math.sin(az[f]), -math.sin(lat)])
             for f in FINGERS}
        obj = {"shape": "sphere", "d": SPH_D, "r": r, "centre": c}
        pc = {f: c + r * n[f] for f in FINGERS}
        p_axis, u_axis = c, np.array([0.0, 0.0, 1.0])
    rr = K.R + K.GAP
    tg = {f: pc[f] + rr * n[f] for f in FINGERS}
    return obj, tg, n, pc, (p_axis, u_axis)


def bounds(fp):
    if fp == "pinch":
        hl = CYL_L / 2 - 0.010
        return [(-0.07, 0.07), (-0.07, 0.07), (CYL_D / 2 + 0.015, 0.110), (-math.radians(35), math.radians(35)),
                (-hl, hl), (-hl, hl), (0.0, math.radians(20)), (0.0, math.radians(20))]
    off = math.radians(50)
    return [(-0.07, 0.07), (-0.07, 0.07), (SPH_D / 2 + 0.010, 0.110), (-off, off), (-off, off), (-off, off),
            (-math.radians(10), math.radians(20))]


def initial(fp, d):
    """A starting guess from the layout alone: the object under the mounts' centroid, contacts facing the mounts."""
    M = mounts6(d)
    cen = np.mean([M[f] for f in FINGERS], axis=0)
    if fp == "pinch":
        sp = 0.5 * (M["thumb"][1] + M["index"][1]) - cen[1]
        return np.array([0.5 * (M["thumb"][0] + M["index"][0]), cen[1], 0.060, 0.0,
                         float(np.clip(sp + 0.015, -0.035, 0.035)), float(np.clip(sp - 0.030, -0.04, 0.04)),
                         math.radians(10), math.radians(5)])
    return np.array([cen[0], cen[1], 0.060, 0.0, 0.0, 0.0, 0.0])


def grid_start(fp, d):
    """The most interior point of a coarse grid over the pose variables (vectorised; the HOM task design's scan), as
    the SQP's start when no neighbour's solution is at hand. Returns (v, its smallest scaled slack)."""
    M = mounts6(d)
    cen = np.mean([M[f] for f in FINGERS], axis=0)
    xs = cen[0] + np.linspace(-0.030, 0.030, 7)
    ys = cen[1] + np.linspace(-0.030, 0.030, 7)
    hs = np.linspace(0.035, 0.095, 7)
    if fp == "pinch":
        r, rr = CYL_D / 2, K.R + K.GAP
        hl = CYL_L / 2 - 0.010
        psis = np.radians([-20.0, 0.0, 20.0])
        ss = np.linspace(-hl, hl, 9)
        phs = np.radians([0.0, 10.0, 20.0])
        X, Y, H, PS, S, PH = np.meshgrid(xs, ys, hs, psis, ss, phs, indexing="ij")
        a = np.stack([-np.sin(PS), np.cos(PS), np.zeros_like(PS)], -1)
        e = np.stack([np.cos(PS), np.sin(PS), np.zeros_like(PS)], -1)
        zz = np.zeros_like(a)
        zz[..., 2] = 1.0
        c = np.stack([X, Y, -H], -1)
        sl = {}
        for f, side in (("thumb", -1.0), ("index", 1.0), ("middle", 1.0)):
            n = side * np.cos(PH)[..., None] * e - np.sin(PH)[..., None] * zz
            T = c + S[..., None] * a + (r + rr) * n
            best = None
            for b in (1.0, -1.0):
                q, reach, marg = finger_state(f, T, M[f], b)
                face = (K.fk(f, q)["palmar"] * (-n)).sum(-1) - COS_FACE[fp]
                m_ = np.minimum(np.minimum(reach, marg.min(-1)), SLACK_FACE * face)
                best = m_ if best is None else np.maximum(best, m_)
            sl[f] = best                                            # (x, y, h, psi, s, phi)
        ti = np.minimum(sl["thumb"], sl["index"])                   # pinch at s_p, latitude phi_p
        mid = sl["middle"].max(-1)                                  # middle at s_m, best latitude
        lever = ss[:, None] - ss[None, :]                           # s_p - s_m
        tot = np.minimum(ti.max(-1)[..., :, None], mid[..., None, :])
        tot = np.where(lever >= LEVER, np.minimum(tot, SLACK_LEVER * (lever - LEVER)), -np.inf)
        k = np.unravel_index(int(np.argmax(tot)), tot.shape)       # (x, y, h, psi, s_p, s_m)
        ip = int(np.argmax(ti[k[0], k[1], k[2], k[3], k[4]]))
        im = int(np.argmax(sl["middle"][k[0], k[1], k[2], k[3], k[5]]))
        v = np.array([xs[k[0]], ys[k[1]], hs[k[2]], psis[k[3]], ss[k[4]], ss[k[5]], phs[ip], phs[im]])
        return v, float(tot[k])
    lats = np.radians([-5.0, 5.0, 15.0])
    X, Y, H, L = np.meshgrid(xs, ys, hs, lats, indexing="ij")
    tot = np.full(X.shape, np.inf)
    for f in FINGERS:
        az = np.arctan2(M[f][1] - Y, M[f][0] - X)
        n = np.stack([np.cos(L) * np.cos(az), np.cos(L) * np.sin(az), -np.sin(L)], -1)
        T = np.stack([X, Y, -H], -1) + (SPH_D / 2 + K.R + K.GAP) * n
        best = None
        for b in (1.0, -1.0):
            q, reach, marg = finger_state(f, T, M[f], b)
            face = (K.fk(f, q)["palmar"] * (-n)).sum(-1) - COS_FACE[fp]
            m_ = np.minimum(np.minimum(reach, marg.min(-1)), SLACK_FACE * face)
            best = m_ if best is None else np.maximum(best, m_)
        tot = np.minimum(tot, best)
    k = np.unravel_index(int(np.argmax(tot)), tot.shape)
    return np.array([xs[k[0]], ys[k[1]], hs[k[2]], 0.0, 0.0, 0.0, lats[k[3]]]), float(tot[k])


def finger_state(f, T, mount, branch):
    """Joint angles of finger f for pad-centre targets T (..., 3) on a fixed elbow branch, the reach residual
    1 - |c2| (>= 0 when reachable) and the joint margins (..., 6) against the servo limits less MARGIN."""
    s = K.FLEX[f]
    Tm = T - mount
    X, Y, Z = Tm[..., 0], Tm[..., 1], Tm[..., 2]
    rho = np.hypot(Y, Z)
    u, vv = -s * X, rho - K.J
    c2 = (u * u + vv * vv - K.J2 * K.J2 - K.P * K.P) / (2.0 * K.J2 * K.P)
    # the elbow is kept C2_MAX off straight and folded: arccos has an infinite slope at |c2| = 1, which stalls SLSQP
    t2 = branch * np.arccos(np.clip(c2, -C2_MAX, C2_MAX))
    q = np.stack([np.arctan2(Y, -Z), np.arctan2(u, vv) - np.arctan2(K.P * np.sin(t2), K.J2 + K.P * np.cos(t2)), t2], -1)
    lo, hi = LIM[f][:, 0] + MARGIN, LIM[f][:, 1] - MARGIN
    return q, C2_MAX - np.abs(c2), np.concatenate([q - lo, hi - q], -1)


def smin_pad(f, q):
    return np.linalg.svd(K.jac_pad(f, q), compute_uv=False)[..., -1]


# scales that make the constraint slacks comparable: joint margins and the reach residual as they are (rad, and
# 1 - |cos| of the PIP's elbow), the facing slack (a cosine) x 3, the lever slack (m) x 5
SLACK_FACE, SLACK_LEVER, SLACK_OBJ, T_CAP = 3.0, 5.0, 20.0, 0.35
C2_MAX = 0.995


class Problem:
    """One layout and one foundational pose. The SQP finds the pose's most interior point: variables (v, t), maximise t
    subject to every scaled constraint slack >= t (capped at T_CAP), so t >= 0 means the pose is reached. This plays the
    role of the paper's FP-specific sampling cost."""

    def __init__(self, fp, d):
        self.fp, self.d, self.M = fp, np.asarray(d, float), mounts6(d)
        self.branch = {f: 1.0 for f in FINGERS}
        self._x, self._val = None, None

    def pick_branches(self, v):
        _, tg, _, _, _ = geometry(self.fp, v, self.M)
        for f in FINGERS:
            best = None
            for b in (1.0, -1.0):
                _q, reach, marg = finger_state(f, tg[f], self.M[f], b)
                score = min(float(reach), float(marg.min()))
                if best is None or score > best[0]:
                    best = (score, b)
            self.branch[f] = best[1]

    def slacks(self, v):
        if self._x is not None and np.array_equal(v, self._x):
            return self._val
        obj, tg, n, _, _ = geometry(self.fp, v, self.M)
        cons = []
        for f in FINGERS:
            q, reach, marg = finger_state(f, tg[f], self.M[f], self.branch[f])
            cons += [reach, *marg]
            pts = K.fk(f, q)
            cons.append(SLACK_FACE * (float(pts["palmar"] @ (-n[f])) - COS_FACE[self.fp]))
            world = {k: p + self.M[f] for k, p in pts.items() if k != "palmar"}
            cons.append(SLACK_OBJ * (float(K.object_clearance(world, obj, n=5)) - OBJ_CLEAR))
        if self.fp == "pinch":
            cons.append(SLACK_LEVER * (v[4] - v[5] - LEVER))
        self._x, self._val = np.array(v, copy=True), np.array(cons, float)
        return self._val

    def solve(self, v0, maxiter=80):
        self.pick_branches(v0)
        nv = len(v0)
        t0 = min(float(self.slacks(np.asarray(v0, float)).min()), T_CAP)
        x0 = np.append(np.asarray(v0, float), t0)
        bnd = bounds(self.fp) + [(-2.0, T_CAP)]
        x0 = np.clip(x0, [b[0] for b in bnd], [b[1] for b in bnd])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(lambda x: -x[-1], x0, jac=lambda x: np.r_[np.zeros(nv), -1.0], method="SLSQP", bounds=bnd,
                           constraints=[{"type": "ineq", "fun": lambda x: self.slacks(x[:nv]) - x[-1]}],
                           options={"maxiter": maxiter, "ftol": 1e-6})
        best = None
        for v in (res.x[:nv], np.asarray(v0, float)):              # the SQP's point, else the start if it is feasible
            sl = self.slacks(v)
            if sl.min() < -1e-6:
                continue
            clear = clearances(self.fp, v, self.M, self.branch)
            if clear[0] >= FF_CLEAR and clear[1] >= OBJ_CLEAR:
                best = (True, np.array(v, copy=True), res.nit, clear, float(sl.min()))
                break
        if best is None:
            return False, res.x[:nv], res.nit, None, float(self.slacks(res.x[:nv]).min())
        return best


def clearances(fp, v, M, branch):
    obj, tg, _, _, _ = geometry(fp, v, M)
    world = {}
    for f in FINGERS:
        q, _, _ = finger_state(f, tg[f], M[f], branch[f])
        world[f] = {k: p + M[f] for k, p in K.fk(f, q).items() if k != "palmar"}
    ff = float(K.finger_clearance(world))
    ob = float(min(K.object_clearance(world[f], obj) for f in FINGERS))
    return ff, ob


def rot(u, ang):
    """Rotation matrices (..., 3, 3) about unit axis u by angles ang (...)."""
    k = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    a = np.asarray(ang)[..., None, None]
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)


def metrics(fp, v, M, branch):
    """TMR (deg, both directions, capped), MFT (mN m) along it, smallest pad singular value (mm/rad) at the pose."""
    obj, tg, n, pc, (p_ax, u_ax) = geometry(fp, v, M)
    ang = np.radians(np.arange(0.0, TMR_CAP + 1e-9, TMR_STEP))
    out = {}
    tau_max = 0.0
    for sgn, key in ((1.0, "plus"), (-1.0, "minus")):
        Rm = rot(u_ax, sgn * ang)                                          # (A, 3, 3)
        ok = np.ones(len(ang), bool)
        world = {}
        tau = np.zeros(len(ang))
        cen = (Rm @ (obj["centre"] - p_ax)) + p_ax
        o2 = dict(obj, centre=cen)
        if fp == "pinch":
            o2["axis"] = Rm @ obj["axis"]
        for f in FINGERS:
            T = (Rm @ (tg[f] - p_ax)) + p_ax                               # pad centres carried with the object
            nn = Rm @ n[f]
            q, reach, marg = finger_state(f, T, M[f], branch[f])
            pts = K.fk(f, q)
            face = (pts["palmar"] * (-nn)).sum(-1)
            ok &= (reach >= 0) & (marg.min(-1) >= -MARGIN) & (face >= FACE_PATH[fp])
            world[f] = {k: p + M[f] for k, p in pts.items() if k != "palmar"}
            F = F_TRIPOD if fp == "tripod" else (F_DRIVE if f == "middle" else F_PINCH)
            Jp = K.jac_pad(f, q)                                           # (A, 3, 3)
            t = np.einsum("aij,ai->aj", Jp, -F * nn)                       # J^T f, f pressing into the object
            tau = np.maximum(tau, np.abs(t).max(-1))
        ok &= K.finger_clearance(world) >= 0.0
        ok &= np.min([K.object_clearance(world[f], o2) for f in FINGERS], axis=0) >= 0.0
        bad = np.flatnonzero(~ok)
        k = int(bad[0]) if bad.size else len(ang)
        out[key] = float(np.degrees(ang[k - 1])) if k > 0 else 0.0
        tau_max = max(tau_max, float(tau[:max(k, 1)].max()))
    q0 = {f: finger_state(f, tg[f], M[f], branch[f])[0] for f in FINGERS}
    sm = min(float(smin_pad(f, q0[f])) for f in FINGERS) * 1000.0
    lever = float(v[4] - v[5]) * 1000.0 if fp == "pinch" else None
    return {"tmr_deg": out["plus"] + out["minus"], "tmr_pm": [out["plus"], out["minus"]], "mft_mNm": tau_max * 1000.0,
            "smin_mm": sm, "lever_mm": lever, "q_deg": {f: np.degrees(q0[f]).round(2).tolist() for f in FINGERS}}


# ------------------------------------------------------------------------------------------------ one candidate

def attempt(d, warm=None):
    """Solve both foundational poses for layout d (mm); warm: {fp: v} of the nearest candidate. Returns a row."""
    row = {"d": [round(float(x), 3) for x in d], "fp": {}}
    t0 = time.perf_counter()
    for fp in FPS:
        pr = Problem(fp, d)
        tries = [warm[fp]] if warm else []
        tries.append(grid_start(fp, d)[0])
        rec = None
        for k, v0 in enumerate(tries):
            ok, v, nit, clear, slack = pr.solve(np.asarray(v0, float))
            if ok:
                rec = {"ok": True, "v": [round(float(x), 6) for x in v], "nit": int(nit), "start": k,
                       "slack": round(slack, 4),
                       "ff_mm": round(clear[0] * 1000, 2), "ob_mm": round(clear[1] * 1000, 2),
                       "branch": pr.branch.copy()}
                rec.update(metrics(fp, v, pr.M, pr.branch))
                break
        row["fp"][fp] = rec or {"ok": False}
        if rec is None:
            break                                              # the paper's candidate must reach every FP
    row["ok"] = all(row["fp"].get(fp, {}).get("ok") for fp in FPS)
    row["ms"] = round(1000 * (time.perf_counter() - t0), 1)
    return row


def append(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def standard(d, lo, hi):
    return (np.asarray(d, float) - lo) / (hi - lo) - 0.5


# ------------------------------------------------------------------------------------------------ samplers

def rrt(name, dim, max_cand, max_calls, step, dmin, seed, eff_floor=0.01, window=400):
    """The paper's sampler: a uniform sample in the standardized space, the nearest candidate, a step toward the
    sample; kept as a candidate when it is at least dmin from every candidate and reaches both poses (warm start: the
    nearest candidate's solution). Stops at max_cand candidates, max_calls RRT calls, or when the candidates per call
    over the last `window` calls fall below eff_floor."""
    ROWS.mkdir(parents=True, exist_ok=True)
    path = ROWS / f"{name}.jsonl"
    if path.exists():
        path.unlink()
    lo, hi = (LO2, HI2) if dim == 2 else (LO6, HI6)
    to_d = (lambda z: layout2(*z)) if dim == 2 else (lambda z: z)
    rng = np.random.default_rng(seed)
    z0 = standard(np.array([100.0, 110.0]) if dim == 2 else SEED6, lo, hi)
    root = attempt(to_d(lo + (z0 + 0.5) * (hi - lo)))
    if not root["ok"]:
        raise SystemExit(f"seed layout does not reach the poses: {root}")
    Z, sols = [z0], [{fp: root["fp"][fp]["v"] for fp in FPS}]
    root.update(call=0, z=z0.round(5).tolist(), parent=None, kind="seed")
    append(path, root)
    calls, recent = 0, []
    t0 = time.time()
    while len(Z) < max_cand and calls < max_calls:
        calls += 1
        u = rng.uniform(-0.5, 0.5, dim)
        Za = np.array(Z)
        dist = np.linalg.norm(Za - u, axis=1)
        i = int(dist.argmin())
        z = u if dist[i] <= step else Za[i] + step * (u - Za[i]) / dist[i]
        far = np.linalg.norm(Za - z, axis=1).min() >= dmin
        got = False
        if far:
            row = attempt(to_d(lo + (z + 0.5) * (hi - lo)), warm=sols[i])
            row.update(call=calls, z=z.round(5).tolist(), parent=i, kind="rrt")
            append(path, row)
            if row["ok"]:
                Z.append(z)
                sols.append({fp: row["fp"][fp]["v"] for fp in FPS})
                got = True
        recent.append(got)
        if len(recent) > window:
            recent.pop(0)
        if calls % 200 == 0:
            print(f"{name}: {calls} calls, {len(Z)} candidates, efficiency {np.mean(recent):.3f}, "
                  f"{time.time() - t0:.0f} s", flush=True)
        if calls >= window and np.mean(recent) < eff_floor:
            break
    summary = {"name": name, "dim": dim, "calls": calls, "candidates": len(Z), "step": step, "dmin": dmin,
               "seed": seed, "efficiency_last": float(np.mean(recent)), "seconds": round(time.time() - t0, 1)}
    (ROWS / f"{name}_summary.json").write_text(json.dumps(summary, indent=1))
    print(summary, flush=True)


def sobol(n, seed, shard=0, shards=1, name="sobol6d"):
    """The same two poses on a scrambled Sobol draw over the gantry box, each from the layout-only guess (no tree).
    With shards > 1, draws k, k + shards, ... go to rows/<name>_shard<k>.jsonl."""
    from scipy.stats import qmc
    ROWS.mkdir(parents=True, exist_ok=True)
    path = ROWS / (f"{name}_shard{shard}.jsonl" if shards > 1 else f"{name}.jsonl")
    if path.exists():
        path.unlink()
    z = qmc.Sobol(6, scramble=True, seed=seed).random(n) - 0.5
    t0 = time.time()
    good = 0
    for k, zz in enumerate(z):
        if k % shards != shard:
            continue
        row = attempt(LO6 + (zz + 0.5) * (HI6 - LO6))
        row.update(call=k, z=zz.round(5).tolist(), kind="sobol")
        append(path, row)
        good += row["ok"]
        if (k + 1) % 200 == 0:
            print(f"sobol: {k + 1} draws, {good} reach both poses, {time.time() - t0:.0f} s", flush=True)
    summary = {"name": name, "n": n, "seed": seed, "shard": shard, "shards": shards, "candidates": good,
               "seconds": round(time.time() - t0, 1)}
    (ROWS / (f"{name}_shard{shard}_summary.json" if shards > 1 else f"{name}_summary.json")).write_text(
        json.dumps(summary, indent=1))
    print(summary, flush=True)


# ------------------------------------------------------------------------------------------------ analysis

KEYS = (("tmr_deg", +1), ("mft_mNm", -1), ("smin_mm", +1))


def load(name):
    rows = [json.loads(x) for x in (ROWS / f"{name}.jsonl").read_text().splitlines() if x.strip()]
    return rows


def scores(rows):
    """Candidates' nine... here six raw metrics (2 poses x 3) and their 0-100 scores (worst 0, best 100)."""
    cand = [r for r in rows if r["ok"]]
    raw = np.array([[r["fp"][fp][k] for fp in FPS for k, _ in KEYS] for r in cand], float)
    sgn = np.array([s for _fp in FPS for _k, s in KEYS], float)
    lo, hi = raw.min(0), raw.max(0)
    sc = np.where(sgn > 0, (raw - lo) / np.maximum(hi - lo, 1e-12), (hi - raw) / np.maximum(hi - lo, 1e-12)) * 100
    return cand, raw, sc


def pareto(sc):
    """Indices of the non-dominated rows of a score matrix (all maximised)."""
    n = len(sc)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(sc >= sc[i], axis=1) & np.any(sc > sc[i], axis=1)
        if dom.any():
            keep[i] = False
    return np.flatnonzero(keep)


def embed():
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    rows = load("rrt6d")
    cand, raw, sc = scores(rows)
    Z = np.array([r["z"] for r in cand])
    out = {"n": len(cand)}
    for perp in (10, 30, 60):
        out[f"tsne_p{perp}"] = TSNE(2, perplexity=perp, init="pca", random_state=0).fit_transform(Z).round(4).tolist()
    sob = ROWS / "sobol6d.jsonl"
    if sob.exists():                     # the same embedding of the Sobol candidates: islands that the tree's branches
        cs = [r for r in load("sobol6d") if r["ok"]]          # make, and not the feasible set, are absent here
        Zs = np.array([r["z"] for r in cs])
        out["sobol_n"] = len(cs)
        out["sobol_tsne_p30"] = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Zs).round(4).tolist()
        sub = np.random.default_rng(0).choice(len(Z), size=min(len(Z), len(cs)), replace=False)
        out["rrt_sub_idx"] = sub.tolist()                       # the RRT candidates at the Sobol count
        out["rrt_sub_tsne_p30"] = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Z[sub]).round(4).tolist()
    pca = PCA(6).fit(Z)
    out["pca"] = pca.transform(Z)[:, :2].round(4).tolist()
    out["pca_var"] = pca.explained_variance_ratio_.round(4).tolist()
    out["pca_axes"] = pca.components_[:2].round(4).tolist()
    (ROWS / "embed6d.json").write_text(json.dumps(out))
    print({k: (v if k in ("n", "pca_var") else "...") for k, v in out.items()}, flush=True)


def embed_sobol8k():
    """t-SNE of the Sobol-8k layouts that reach both poses (rows/sobol8k.jsonl, the merged shards): the unbiased map."""
    from sklearn.manifold import TSNE
    rows = [r for r in load("sobol8k") if r["ok"]]
    Z = np.array([r["z"] for r in rows])
    Y = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Z)
    (ROWS / "embed_sobol8k.json").write_text(json.dumps({"d": [r["d"] for r in rows], "tsne": Y.round(4).tolist()}))
    print("embedded", len(rows), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p2 = sub.add_parser("run2d")
    p2.add_argument("--max", type=int, default=1500)
    p2.add_argument("--calls", type=int, default=60000)
    p6 = sub.add_parser("run6d")
    p6.add_argument("--max", type=int, default=2500)
    p6.add_argument("--calls", type=int, default=20000)
    ps = sub.add_parser("sobol")
    ps.add_argument("--n", type=int, default=1024)
    ps.add_argument("--shard", type=int, default=0)
    ps.add_argument("--shards", type=int, default=1)
    ps.add_argument("--name", default="sobol6d")
    pt = sub.add_parser("one")
    pt.add_argument("d", type=float, nargs=6)
    sub.add_parser("embed")
    sub.add_parser("embed8k")
    a = ap.parse_args()
    if a.cmd == "run2d":
        rrt("rrt2d", 2, a.max, a.calls, step=0.05, dmin=0.03, seed=1)
    elif a.cmd == "run6d":
        rrt("rrt6d", 6, a.max, a.calls, step=0.10, dmin=0.04, seed=2)
    elif a.cmd == "sobol":
        sobol(a.n, seed=3, shard=a.shard, shards=a.shards, name=a.name)
    elif a.cmd == "one":
        print(json.dumps(attempt(np.array(a.d)), indent=1))
    elif a.cmd == "embed":
        embed()
    elif a.cmd == "embed8k":
        embed_sobol8k()


if __name__ == "__main__":
    main()
