"""Kinematics of the Universal Robots arms in closed form: the flange's pose from the joints, every joint solution of a
flange pose, and one fixed choice among them.

Units are millimetres and radians. B is the controller's base frame ('base' in ur_description's URDF, turned half a
turn about z from 'base_link'); E is the flange, tool0.

The arm is data (URParams): the standard Denavit-Hartenberg table that Universal Robots publishes, with joint i
turning by q_i and T_i = Rot_z(q_i) Trans_z(d_i) Trans_x(a_i) Rot_x(alpha_i). UR5_CB3 is the UR5's nominal table, the
one of the lab's simulator (ur5_tip_calibration_sim.DH_CB3), of ur_description's ur5.urdf and of the cad_v3 MuJoCo
scenes. Another UR arm loads with URParams.from_dict or URParams.from_json. The closed form needs the UR geometry (a_1
= a_4 = a_5 = a_6 = 0, d_2 = d_3 = 0, alpha = (pi/2, 0, 0, pi/2, -pi/2, 0)); URParams refuses a table without it, such
as a controller's calibrated table (its calibration.conf deltas leave joints 2 to 4 not exactly parallel).

The solution follows K. P. Hawkins, "Analytic inverse kinematics for the Universal Robots UR-5/UR-10 arms" (Georgia
Tech, 2013), and is the arithmetic of scripts/ur5/ur5_inverted_rig.py's ik_all of 2026-10-29, which moved here on
2026-11-05. A pose has up to eight solutions, one per branch: shoulder s1 (joint 1 = psi + s1 phi + pi/2), wrist s5
(joint 5 = s5 acos c5) and elbow s3 (joint 3 = s3 acos c3), in the fixed order of BRANCHES:

    k   0  1  2  3  4  5  6  7
    s1  +  +  +  +  -  -  -  -
    s5  +  +  -  -  +  +  -  -
    s3  +  -  +  -  +  -  +  -

ik() also says why a branch has no solution (UNREACH_*: the wrist's centre inside joint 1's cylinder of radius d4, the
wrist, or the elbow out of reach) and which branches lie at a singularity (SING_*: the two shoulder branches meet,
joints 4 and 6 line up at sin q5 = 0, the arm stretched or folded at sin q3 = 0). Within WRIST_NEAR of the wrist
singularity, joint 5 comes from atan2 of sin q5 (from the flange's axes) and c5, since acos(c5) keeps sin q5 to only
about 1e-8 there. At the singularity itself, |sin q5| under WRIST_FLOOR, joint 6 is free and takes the value q6_des (0
by default; ik_near takes the reference's joint 6), as in Hawkins' code, and joints 2 to 4 are solved for it; the pose
is then reached when that joint 6 lets the elbow reach.

The choice (ik_near, ik_nearest): each solution's joints are taken to the equivalent nearest the reference joints
q_ref (numerics.angles.nearest, within half a turn), optionally moved by a whole turn into the arm's joint limits; the
solution with the least Euclidean joint distance to q_ref is kept, and solutions within TIE_TOL rad of the least
distance go to the lowest k. The same pose and reference always give the same solution. A pose changed by 1e-12 gives
the same branch unless the runner-up's distance lies within about 1e-12 rad of the least distance plus TIE_TOL. Its
joints then move by the change times the arm's conditioning (more near a singularity), or by a whole turn where a
joint lies half a turn from q_ref's.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from sr2hand.geometry.rotations import project_to_so3, tf_inv
from sr2hand.numerics.angles import nearest, wrap

__all__ = [
    "URParams", "UR5_CB3", "BRANCHES", "IKSolutions", "UNREACH_SHOULDER", "UNREACH_WRIST", "UNREACH_ELBOW",
    "SING_SHOULDER", "SING_WRIST", "SING_ELBOW", "REACH_TOL", "SINGULAR_TOL", "TIE_TOL",
    "dh", "fk", "fk_many", "ik", "ik_all", "branch_of", "wrap_to", "rank", "ik_near", "ik_nearest",
]

TWO_PI = 2.0 * math.pi
UR_ALPHA = (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)
BRANCHES = tuple((s1, s5, s3) for s1 in (1, -1) for s5 in (1, -1) for s3 in (1, -1))
UNREACH_SHOULDER, UNREACH_WRIST, UNREACH_ELBOW = 1, 2, 4
SING_SHOULDER, SING_WRIST, SING_ELBOW = 1, 2, 4
REACH_TOL = 1e-12        # |cos| up to 1 + REACH_TOL counts as reached (rounding at a stretched arm or a lined-up wrist)
SINGULAR_TOL = 1e-6      # rad: phi (shoulder), |sin q5| (wrist) and |sin q3| (elbow) under this flag a singularity
TIE_TOL = 1e-9           # rad: joint distances this close count as equal, and the lower branch index wins
WRIST_NEAR = 1e-6        # |sin q5| under this: sin q5 from the flange's axes (acos(c5) is 2e-10 rad off at 1e-6)
WRIST_FLOOR = 1e-12      # |sin q5| under this: joints 4 and 6 lined up, joint 6 set to q6_des


@dataclass(frozen=True)
class URParams:
    """A UR arm's standard DH table (mm, rad) and its joint limits (rad)."""
    name: str
    d: tuple
    a: tuple
    alpha: tuple = UR_ALPHA
    q_min: tuple = (-TWO_PI,) * 6
    q_max: tuple = (TWO_PI,) * 6
    source: str = ""

    def __post_init__(self):
        for k in ("d", "a", "alpha", "q_min", "q_max"):
            v = tuple(float(x) for x in getattr(self, k))
            if len(v) != 6:
                raise ValueError(f"{self.name}: {k} has {len(v)} entries, a UR arm 6")
            object.__setattr__(self, k, v)
        d, a, al = self.d, self.a, self.alpha
        if a[0] or a[3] or a[4] or a[5] or d[1] or d[2] or al != UR_ALPHA:
            raise ValueError(f"{self.name}: not a UR table (a_1 = a_4 = a_5 = a_6 = 0, d_2 = d_3 = 0, alpha = "
                             f"(pi/2, 0, 0, pi/2, -pi/2, 0)); the closed form needs that geometry, and a calibrated "
                             f"table lacks it")
        if not (d[5] and a[1] and a[2]):
            raise ValueError(f"{self.name}: d_6, a_2 and a_3 must be nonzero")

    @classmethod
    def from_dict(cls, rec):
        """A table from a dict with name, d and a (and optionally alpha, q_min, q_max, source); units 'mm' (default) or
        'm' scale d and a to mm."""
        s = {"mm": 1.0, "m": 1000.0}[rec.get("units", "mm")]
        kw = {k: rec[k] for k in ("alpha", "q_min", "q_max", "source") if k in rec}
        return cls(name=rec["name"], d=[float(v) * s for v in rec["d"]], a=[float(v) * s for v in rec["a"]], **kw)

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self):
        return dict(asdict(self), units="mm")


UR5_CB3 = URParams(
    name="UR5 CB3, nominal", d=(89.159, 0.0, 0.0, 109.15, 94.65, 82.3), a=(0.0, -425.0, -392.25, 0.0, 0.0, 0.0),
    source="Universal Robots' DH table of the UR5; ur_description's ur5.urdf; ur5_tip_calibration_sim.DH_CB3. Every "
           "joint +-360 deg (CB3)")


@dataclass
class IKSolutions:
    """Every solution of flange poses T (..., 4, 4): q (..., 8, 6) in BRANCHES' order, angles in [-pi, pi) and NaN where
    the branch has none; ok (..., 8); unreachable and singular (..., 8), bit masks of UNREACH_* and SING_*."""
    q: np.ndarray
    ok: np.ndarray
    unreachable: np.ndarray
    singular: np.ndarray = field(repr=False)


# ------------------------------------------------------------------------------------------------ forward kinematics
def dh(i, th, P=UR5_CB3):
    """The DH transform of joint i (0..5) at angles th (...), shape (..., 4, 4)."""
    th = np.asarray(th, float)
    ct, st = np.cos(th), np.sin(th)
    ca, sa = math.cos(P.alpha[i]), math.sin(P.alpha[i])
    A = np.zeros(th.shape + (4, 4))
    A[..., 0, 0], A[..., 0, 1], A[..., 0, 2], A[..., 0, 3] = ct, -st * ca, st * sa, P.a[i] * ct
    A[..., 1, 0], A[..., 1, 1], A[..., 1, 2], A[..., 1, 3] = st, ct * ca, -ct * sa, P.a[i] * st
    A[..., 2, 1], A[..., 2, 2], A[..., 2, 3] = sa, ca, P.d[i]
    A[..., 3, 3] = 1.0
    return A


def fk_many(Q, P=UR5_CB3):
    """Flange poses T_BE (N, 4, 4) of joints Q (N, 6)."""
    Q = np.atleast_2d(Q)
    T = dh(0, Q[:, 0], P)
    for i in range(1, 6):
        T = T @ dh(i, Q[:, i], P)
    return T


def fk(q, P=UR5_CB3):
    """The flange's pose T_BE of joints q: (4, 4) for q of shape (6,), (N, 4, 4) for (N, 6)."""
    q = np.asarray(q, float)
    T = fk_many(q.reshape(-1, 6), P)
    return T[0] if q.ndim == 1 else T


# ------------------------------------------------------------------------------------------------ inverse kinematics
def _solve(T, P, q6_des=0.0):
    """Q (N, 8, 6) and the flags of flange poses T (N, 4, 4). Away from the wrist singularity the arithmetic of every
    reached branch is that of ur5_inverted_rig.ik_all of 2026-10-29; the reach tests take REACH_TOL of rounding, and at
    the wrist singularity joint 6 is q6_des (a number or one per pose), and within WRIST_NEAR of it joint 5 comes from
    atan2."""
    T = np.asarray(T, float).reshape(-1, 4, 4)
    N = len(T)
    q6_des = np.broadcast_to(np.asarray(q6_des, float), (N,))
    d4, d6, a2, a3 = P.d[3], P.d[5], P.a[1], P.a[2]
    Q = np.full((N, 8, 6), np.nan)
    unreach = np.zeros((N, 8), np.uint8)
    sing = np.zeros((N, 8), np.uint8)
    R, p = T[:, :3, :3], T[:, :3, 3]
    p05 = p - d6 * R[:, :, 2]                 # joint 5's axis meets joint 6's here; joint 1 sees it at distance r
    r = np.hypot(p05[:, 0], p05[:, 1])
    ok1 = r >= abs(d4) - 1e-9                 # within 1e-9 mm of the cylinder: the shoulder branches meet (phi = 0)
    phi = np.arccos(np.clip(abs(d4) / np.maximum(r, 1e-9), -1.0, 1.0))
    psi = np.arctan2(p05[:, 1], p05[:, 0])
    k = 0
    for s1 in (1.0, -1.0):
        q1 = psi + s1 * phi + math.pi / 2
        S1, C1 = np.sin(q1), np.cos(q1)
        c5 = (p[:, 0] * S1 - p[:, 1] * C1 - d4) / d6
        ok5 = ok1 & (np.abs(c5) <= 1.0 + REACH_TOL)
        a5 = np.arccos(np.clip(c5, -1.0, 1.0))
        for s5 in (1.0, -1.0):
            q5 = s5 * a5
            S5 = np.sin(q5)
            near = np.abs(S5) < WRIST_NEAR
            if near.any():                  # acos(c5) keeps sin q5 to about 1e-8 at the singularity; the axes, to 1e-16
                m = np.hypot(R[:, 0, 0] * S1 - R[:, 1, 0] * C1, R[:, 0, 1] * S1 - R[:, 1, 1] * C1)
                q5 = np.where(near, s5 * np.arctan2(m, c5), q5)
                S5 = np.where(near, np.sin(q5), S5)
            sing_w = np.abs(S5) < SINGULAR_TOL
            free = np.abs(S5) < WRIST_FLOOR                     # joints 4 and 6 lined up: joint 6 is free
            S5 = np.where(free, 1.0, S5)
            q6 = np.arctan2((-R[:, 0, 1] * S1 + R[:, 1, 1] * C1) / S5, (R[:, 0, 0] * S1 - R[:, 1, 0] * C1) / S5)
            q6 = np.where(free, q6_des, q6)
            T14 = tf_inv(dh(0, q1, P)) @ T @ tf_inv(dh(4, q5, P) @ dh(5, q6, P))
            x, y = T14[:, 0, 3], T14[:, 1, 3]
            c3 = (x * x + y * y - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
            ok3 = ok5 & (np.abs(c3) <= 1.0 + REACH_TOL)
            b3 = np.arccos(np.clip(c3, -1.0, 1.0))
            for s3 in (1.0, -1.0):
                q3 = s3 * b3
                q2 = np.arctan2(y, x) - np.arctan2(a3 * np.sin(q3), a2 + a3 * np.cos(q3))
                T34 = tf_inv(dh(1, q2, P) @ dh(2, q3, P)) @ T14
                q4 = np.arctan2(T34[:, 1, 0], T34[:, 0, 0])
                sol = np.stack([q1, q2, q3, q4, q5, q6], axis=1)
                sol[~ok3] = np.nan
                Q[:, k] = sol
                unreach[:, k] = np.where(~ok1, UNREACH_SHOULDER, np.where(~ok5, UNREACH_WRIST,
                                                                          np.where(~ok3, UNREACH_ELBOW, 0)))
                sing[:, k] = (np.where(phi < SINGULAR_TOL, SING_SHOULDER, 0)
                              | np.where(sing_w, SING_WRIST, 0)
                              | np.where(np.abs(np.sin(b3)) < SINGULAR_TOL, SING_ELBOW, 0)) * ok3
                k += 1
    return Q, unreach, sing


def ik_all(T, P=UR5_CB3, q6_des=0.0):
    """Every solution (N, 8, 6) of flange poses T (N, 4, 4) or (4, 4), in BRANCHES' order; NaN where a branch has none.
    T's rotations are taken as they are: one off orthonormal by e gives solutions off its pose by about e (and e d6 mm).
    The angles are the formulas' own, not wrapped: joint 1 within (-pi, 2 pi], joint 2 within (-2 pi, 2 pi), the rest
    within [-pi, pi] (joint 6 at the wrist singularity: q6_des)."""
    return _solve(T, P, q6_des)[0]


def ik(T, P=UR5_CB3, q6_des=0.0):
    """Every solution of flange poses T (4, 4) or (N, 4, 4), with why a branch has none and where it is singular."""
    T = np.asarray(T, float)
    Q, un, sg = _solve(T, P, q6_des)
    Q = wrap(Q)
    if T.ndim == 2:
        Q, un, sg = Q[0], un[0], sg[0]
    return IKSolutions(q=Q, ok=np.isfinite(Q).all(axis=-1), unreachable=un, singular=sg)


def branch_of(q, P=UR5_CB3):
    """The branch k (0..7, BRANCHES' order) that joints q lie on: the solution of fk(q) nearest q by the largest wrapped
    joint difference (ur5_inverted_rig.branch_of)."""
    Q = ik_all(fk_many(np.asarray(q, float)[None], P), P)[0]
    d = np.abs(wrap(Q - q)).max(axis=1)
    return int(np.nanargmin(np.where(np.isnan(d), np.inf, d)))


# ------------------------------------------------------------------------------------------------ the choice
def wrap_to(Q, q_ref, P=UR5_CB3, limits=True):
    """Each joint of the solutions Q (..., 6) as its equivalent nearest q_ref (within half a turn); with limits, a joint
    outside P's limits moves by a whole turn toward them, and a joint that no turn brings inside becomes NaN."""
    Qw = nearest(Q, q_ref)
    if limits:
        lo, hi = np.asarray(P.q_min), np.asarray(P.q_max)
        Qw = np.where(Qw > hi, Qw - TWO_PI, Qw)
        Qw = np.where(Qw < lo, Qw + TWO_PI, Qw)
        Qw = np.where((Qw < lo) | (Qw > hi), np.nan, Qw)
    return Qw


def rank(T, q_ref, P=UR5_CB3, free6=False, limits=True, branch=None):
    """The choice for flange poses T (N, 4, 4): (Qw (N, 8, 6) wrapped to q_ref, dist (N, 8) with inf where a branch has
    none, best (N,), the runner-up's distance (N,)). dist is the Euclidean joint distance to q_ref over joints 1 to 6
    (1 to 5 with free6, for a tool that may turn about its own axis); branch keeps one branch only."""
    T = np.asarray(T, float).reshape(-1, 4, 4)
    qr = np.asarray(q_ref, float).reshape(-1, 1, 6)
    Q = ik_all(T, P, q6_des=np.broadcast_to(qr[:, 0, 5], (len(T),)))
    Qw = wrap_to(Q, qr, P, limits)
    dq = (Qw - qr)[..., :5 if free6 else 6]
    dist = np.sqrt(np.sum(dq * dq, axis=-1))
    if branch is not None:
        keep = np.zeros(8, bool)
        keep[branch] = True
        dist = np.where(keep[None, :], dist, np.inf)
    dist = np.where(np.isnan(dist), np.inf, dist)
    dmin = dist.min(axis=1, keepdims=True)
    best = np.argmax(dist <= dmin + TIE_TOL, axis=1)          # the lowest k within TIE_TOL of the least distance
    rest = np.where(np.arange(8)[None, :] == best[:, None], np.inf, dist)
    return Qw, dist, best, rest.min(axis=1)


def ik_near(T, q_ref, tol=0.01, free6=False, branch=None, P=UR5_CB3, limits=False, tol_rad=None):
    """The chosen solution (N, 6) for each flange pose of T (N, 4, 4), NaN where none exists or where its forward
    kinematics stands more than tol mm (and, given tol_rad, tol_rad rad) off the pose. The defaults are those of
    ur5_inverted_rig.ik_near, whose replays call it with a branch: joints wrapped to q_ref without the limits."""
    T = np.asarray(T, float).reshape(-1, 4, 4)
    Qw, dist, best, _ = rank(T, q_ref, P, free6, limits, branch)
    out = Qw[np.arange(len(T)), best]
    bad = ~np.isfinite(dist[np.arange(len(T)), best])
    good = ~bad
    if good.any():
        F = fk_many(out[good], P)
        err = np.linalg.norm(F[:, :3, 3] - T[good, :3, 3], axis=1)
        if tol_rad is not None:
            Rd = np.swapaxes(F[:, :3, :3], 1, 2) @ T[good, :3, :3]
            c = np.clip((np.trace(Rd, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
            ang = np.arctan2(np.linalg.norm(Rd[:, [2, 0, 1], [1, 2, 0]] - Rd[:, [1, 2, 0], [2, 0, 1]], axis=1) / 2.0, c)
            err_bad = (err > tol) | (ang > tol_rad)
        else:
            err_bad = err > tol
        bad[np.flatnonzero(good)[err_bad]] = True
    out[bad] = np.nan
    return out


def ik_nearest(T, q_ref, P=UR5_CB3, free6=False, branch=None, tol=1e-4, tol_rad=1e-6):
    """The plan builders' inverse kinematics: the solution of flange pose T (4, 4) nearest the joints q_ref (6,) within
    the arm's joint limits (ik_near's choice), checked against the forward kinematics to tol mm and tol_rad rad; None
    when the pose has none. It takes the place of the simulator's damped least squares (ur5_tip_calibration_sim.ik),
    which on some poses returned another solution when the pose changed in its last bits, and keeps its acceptance
    (1e-4 mm, 1e-6 rad). T's rotation is first taken to the nearest rotation (rotations.project_to_so3), as the least
    squares took its error through the same projection: plans and registrations keep rotations rounded to 4 to 7
    decimals, up to 8e-5 off orthonormal, where the closed form would miss the position by up to d6 times that."""
    T = np.array(T, float)
    T[:3, :3] = project_to_so3(T[:3, :3])
    q = ik_near(T[None], q_ref, tol=tol, free6=free6, branch=branch, P=P, limits=True, tol_rad=tol_rad)[0]
    return None if not np.isfinite(q).all() else q
