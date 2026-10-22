"""Relative contact-velocity control of a multi-finger hand: the hand-object mechanism (HOM) framework of
S. Wang, J. Oh and N. S. Pollard, "Relative Contact Velocity-Controlled Hand-Object Mechanism for Dexterous Tool
Manipulation", arXiv 2609.25619v2, implemented for MuJoCo hands driven by position servos.

Paper to code (equation numbers are the paper's):

  Sec. III-B, generalized contact frames (GCFs)      Gcf, Mirror.gcf_witness, Mirror.gcf_from_cop
  Eq. (1), relative contact twist                    HomController.eq1_rows
  Eq. (2), least squares of the contact velocities   solve_eq2
  Sec. III-B, pinch and pusher sub-assemblies         FingerCommand.mode, HandCommand.pinch
  Eq. (3), reduced-order pinch speed s               pusher_body_twist
  Sec. III-C, anti-windup of v_sep near contact      HomController.step (aw_dist)
  Sec. III-D, Eqs. (4)-(6), complementary filter     ContactFilter
  Sec. III-D, roll-slide contact kinematics          surface_jacobian, roll_slide_rates
  Sec. III-C/D, impedance controller                 servo offset J^T (F x_C1) / k_p in HomController.step

Conventions are the paper's. Each finger's GCF {L1} has its x axis along the finger body's outward normal, pointing
toward the object, so a positive v_sep closes the gap; y is the finger body's joint axis projected on the tangent plane
(maximally aligned with it); z = x cross y. The object's GCF {L0} has x0 = -x1 and y0 = y1. A relative twist is ordered
(w_x, w_y, w_z, v_x, v_y, v_z) = (spin, roll_y, roll_z, sep, slide_y, slide_z): the twist of {L1} relative to the
object, at {L1}'s origin, in {L1}'s axes. Object twists are body twists (w_b, v_b) in the object frame {B0}.

Changes for MuJoCo and for servo-driven hardware (documented in docs/experiments/20261021-hom_framework/):
  * The joints are position servos (MuJoCo position actuators with the bench-identified gain k_p), not torque-controlled
    joints under inverse dynamics. The integrated joint reference theta_ref is the servo target. The impedance
    controller becomes two servo offsets for a commanded force F: J^+ (gap x_C1), which moves the action point onto the
    object's surface, and J^T (F x_C1) / k_p, which presses with F through the servo spring; plus an optional gravity
    feedforward tau_g / k_p.
  * Witness points come from mj_geomDistance on a kinematic copy of the scene (the Mirror) instead of FCL/fcpw on
    convex decompositions; in contact, the centre of pressure of the sphere-pad contacts plays the role of the
    hydroelastic centre of pressure, and the closest points come from analytic surfaces (surfaces.py) instead of
    meshes with libigl curvatures.
  * Linear rows of Eq. (2) are divided by ELL = 20 mm so that 20 mm/s weighs like 1 rad/s; the paper does not give
    its weights or units.
  * theta_ref stays within LEAD of the measured angle, so a blocked finger does not wind its reference up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .surfaces import Posed, unit

COMPONENTS = ("spin", "roll_y", "roll_z", "sep", "slide_y", "slide_z")
SEP = 3
ELL = 0.02                       # m; linear rows of Eq. (2) are divided by ELL
SCALE = np.array([1.0, 1.0, 1.0, 1.0 / ELL, 1.0 / ELL, 1.0 / ELL])


# ------------------------------------------------------------------------------------------------ specs and frames

@dataclass
class FingerSpec:
    name: str
    joints: list[str]                   # joint names, in the order of this finger's slice of q
    body: str                           # the finger body that touches the object (the distal link)
    geom: str                           # its convex collision geom in the Mirror (witness points)
    axis_joint: str                     # joint whose axis sets the GCF y axis (the distal joint)
    surface: Posed | None = None        # analytic surface of the touching body, in its body frame


@dataclass
class ObjectSpec:
    body: str
    geom: str
    surface: Posed | None = None        # analytic surface of the object, in its body frame


@dataclass
class Gcf:
    """The generalized contact frames of one finger (paper Sec. III-B), in world coordinates."""
    dist: float                         # signed distance between the bodies along x1 (negative: penetration)
    p1: np.ndarray                      # origin of {L1}, on the finger
    p0: np.ndarray                      # origin of {L0}, on the object
    R1: np.ndarray                      # columns x1, y1, z1
    source: str                         # "witness" (bodies apart, Mirror) or "cop" (centre of pressure)

    @property
    def x1(self):
        return self.R1[:, 0]

    @property
    def R0(self):
        x0, y0 = -self.R1[:, 0], self.R1[:, 1]
        return np.column_stack([x0, y0, np.cross(x0, y0)])


def frame_from_normal(x: np.ndarray, axis: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Columns (x, y, z): y is `axis` projected on the plane normal to x, z = x cross y."""
    y = axis - (axis @ x) * x
    if np.linalg.norm(y) < 1e-6:
        y = fallback - (fallback @ x) * x
    y = unit(y)
    return np.column_stack([x, y, np.cross(x, y)])


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def skew(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def adjoint(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Ad_T for T = (R, p), acting on twists ordered (w, v)."""
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[3:, 3:] = R
    A[3:, :3] = skew(p) @ R
    return A


# ------------------------------------------------------------------------------------------------ the Mirror

class Mirror:
    """Kinematic MuJoCo copy of the hand and the object: joint angles and the object pose in, GCFs and Jacobians out.
    The controller never reads the plant's internals, so one controller drives a MuJoCo plant, a Drake plant or the
    bench (joint angles from the servo bus, the object pose from the tag tracker)."""

    def __init__(self, model, fingers: list[FingerSpec], obj: ObjectSpec, distmax: float = 0.3):
        import mujoco
        self.mj = mujoco
        self.m = model
        self.d = mujoco.MjData(model)
        self.fingers = {f.name: f for f in fingers}
        self.order = [f.name for f in fingers]
        self.obj = obj
        self.distmax = distmax
        m = model
        self.qadr = {f.name: np.array([m.jnt_qposadr[m.joint(j).id] for j in f.joints]) for f in fingers}
        self.vadr = {f.name: np.array([m.jnt_dofadr[m.joint(j).id] for j in f.joints]) for f in fingers}
        self.lo = {f.name: np.array([m.jnt_range[m.joint(j).id][0] for j in f.joints]) for f in fingers}
        self.hi = {f.name: np.array([m.jnt_range[m.joint(j).id][1] for j in f.joints]) for f in fingers}
        self.bid = {f.name: m.body(f.body).id for f in fingers}
        self.gid = {f.name: m.geom(f.geom).id for f in fingers}
        self.aid = {f.name: m.joint(f.axis_joint).id for f in fingers}
        self.ob = m.body(obj.body).id
        self.og = m.geom(obj.geom).id if obj.geom else int(np.flatnonzero(m.geom_bodyid == self.ob)[0])
        ja = m.body_jntadr[self.ob]
        self.obj_free = m.body_jntnum[self.ob] > 0 and m.jnt_type[ja] == mujoco.mjtJoint.mjJNT_FREE
        self.oq = m.jnt_qposadr[ja] if self.obj_free else None
        self.jacp, self.jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self.kp = {}
        for f in fingers:
            k = []
            for j in f.joints:
                jid = m.joint(j).id
                acts = [a for a in range(m.nu) if m.actuator_trnid[a][0] == jid]
                k.append(float(m.actuator_gainprm[acts[0]][0]) if acts else float("nan"))
            self.kp[f.name] = np.array(k)

    def set_state(self, q: dict[str, np.ndarray], obj_pos, obj_quat):
        d = self.d
        for f, qa in self.qadr.items():
            d.qpos[qa] = q[f]
        if self.obj_free:
            d.qpos[self.oq:self.oq + 3] = obj_pos
            d.qpos[self.oq + 3:self.oq + 7] = obj_quat
        d.qvel[:] = 0.0
        self.mj.mj_kinematics(self.m, d)
        self.mj.mj_comPos(self.m, d)

    def obj_pose(self):
        return self.d.xpos[self.ob].copy(), self.d.xmat[self.ob].reshape(3, 3).copy()

    def body_pose(self, f):
        b = self.bid[f]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def joint_axis(self, f):
        return self.d.xaxis[self.aid[f]].copy()

    def jac(self, f, p):
        """World Jacobians (3 x n_f) of the finger body's material point at p, over this finger's joints."""
        self.mj.mj_jac(self.m, self.d, self.jacp, self.jacr, p, self.bid[f])
        cols = self.vadr[f]
        return self.jacp[:, cols].copy(), self.jacr[:, cols].copy()

    def gravity(self, f):
        """Joint torques that hold the finger against its weight at the current posture (zero velocity)."""
        out = np.zeros(self.m.nv)
        self.mj.mj_rne(self.m, self.d, 0, out)
        return out[self.vadr[f]]

    def gcf_witness(self, f) -> Gcf:
        """GCF from the witness points of the finger's convex geom and the object (paper: the bodies' closest points;
        x1 along the gradient of their signed distance, pointing from the finger toward the object)."""
        ft = np.zeros(6)
        dist = self.mj.mj_geomDistance(self.m, self.d, self.gid[f], self.og, self.distmax, ft)
        p1, p0 = ft[:3].copy(), ft[3:].copy()
        if dist > 1e-6:
            x1 = unit(p0 - p1)
        else:                                             # touching or overlapping: the object's surface normal
            x1 = -self._obj_normal(p0)
        R1 = frame_from_normal(x1, self.joint_axis(f), np.array([0.0, 0.0, 1.0]))
        return Gcf(float(dist), p1, p0, R1, "witness")

    def _obj_normal(self, p_world):
        o, Ro = self.obj_pose()
        if self.obj.surface is not None:
            return Ro @ self.obj.surface.closest(Ro.T @ (p_world - o))[2]
        return unit(p_world - o)

    def gcf_from_cop(self, f, cop: np.ndarray) -> Gcf:
        """GCF in contact (paper Sec. III-E): the origins are the points of each body's surface closest to the centre
        of pressure; the normals are the bodies' outward normals there. Needs analytic surfaces on both bodies."""
        fs, os_ = self.fingers[f].surface, self.obj.surface
        pb, Rb = self.body_pose(f)
        o, Ro = self.obj_pose()
        _, q1, n1 = fs.closest(Rb.T @ (cop - pb))
        _, q0, n0 = os_.closest(Ro.T @ (cop - o))
        p1, p0 = pb + Rb @ q1, o + Ro @ q0
        x1 = unit(Rb @ n1 - Ro @ n0)                      # the finger's outward normal, averaged with -n_object
        R1 = frame_from_normal(x1, self.joint_axis(f), np.array([0.0, 0.0, 1.0]))
        return Gcf(float((p0 - p1) @ x1), p1, p0, R1, "cop")


# ------------------------------------------------------------------------------------------------ Eq. (2), Eq. (3)

def solve_eq2(A, b, v_ref, w, theta, theta_ref, lo, hi, dt, theta_healthy=None, w_damp=1e-3, w_healthy=1e-5,
              k_healthy=1.0, u_max=2.0, scale=None):
    """Eq. (2) for one finger: u = argmin sum_i w_i |v_i,ref - v_i(u)|^2 + w_damp |u|^2
    + w_healthy |k_p (theta_healthy - theta) - u|^2, subject to the joint limits after one step and |u| <= u_max.
    v(u) = A u + b is Eq. (1) at this finger's joint rates; rows are scaled by SCALE (linear rows by 1/ELL).
    The joint-limit bound is applied to the integrated reference theta_ref (the servo target)."""
    from scipy.optimize import lsq_linear
    n = A.shape[1]
    sw = np.sqrt(np.asarray(w, float)) * (SCALE if scale is None else np.asarray(scale, float))
    rows = [sw[:, None] * A, math.sqrt(w_damp) * np.eye(n)]
    rhs = [sw * (np.asarray(v_ref, float) - b), np.zeros(n)]
    if theta_healthy is not None and w_healthy > 0:
        rows.append(math.sqrt(w_healthy) * np.eye(n))
        rhs.append(math.sqrt(w_healthy) * k_healthy * (np.asarray(theta_healthy) - theta))
    lb = np.maximum(-u_max, (lo - theta_ref) / dt)
    ub = np.minimum(u_max, (hi - theta_ref) / dt)
    lb, ub = np.minimum(lb, ub - 1e-12), np.maximum(ub, lb + 1e-12)
    return lsq_linear(np.vstack(rows), np.concatenate(rhs), bounds=(lb, ub), method="bvls").x


def pusher_body_twist(s: float, R_bm: np.ndarray, p_bm: np.ndarray) -> np.ndarray:
    """Eq. (3): V_PB0 = Ad_{T_B0M} (s, 0, 0, 0, 0, 0): the object's body twist for a rotation at speed s about the
    x axis of the pinch frame {M} (origin at the midpoint of the pinch contacts, x along the line joining them),
    whose pose in the object frame is (R_bm, p_bm)."""
    return adjoint(R_bm, p_bm) @ np.array([s, 0.0, 0.0, 0.0, 0.0, 0.0])


def pinch_frame(p_a: np.ndarray, p_b: np.ndarray, obj_pos: np.ndarray, R_obj: np.ndarray, up=None):
    """{M} in the object frame from the two pinch contacts (world): origin the midpoint, x from contact a to b."""
    x = unit(p_b - p_a)
    up = np.array([0.0, 0.0, 1.0]) if up is None else up
    R = frame_from_normal(x, up, np.array([1.0, 0.0, 0.0]))
    mid = 0.5 * (p_a + p_b)
    return R_obj.T @ R, R_obj.T @ (mid - obj_pos)


# ------------------------------------------------------------------------------------------------ Eqs. (4)-(6)

def surface_jacobian(K: np.ndarray) -> np.ndarray:
    """G(K) (6 x 2): the twist (w, v) of a surface frame (x the outward normal, y and z tangent) relative to its body
    when its origin moves with tangent velocity gdot (components along y and z); K is the 2 x 2 shape operator in the
    (y, z) basis. v = (0, gdot); the normal turns as dn = K gdot, so w_y = -(K gdot)_z and w_z = (K gdot)_y. The spin
    row is left at zero: the paper's S drops it."""
    G = np.zeros((6, 2))
    G[1, :] = -K[1, :]
    G[2, :] = K[0, :]
    G[4, 0] = 1.0
    G[5, 1] = 1.0
    return G


S_ROWS = (1, 2, 4, 5)            # the paper's S: the non-normal elements (roll_y, roll_z, slide_y, slide_z)


def roll_slide_rates(V_ref: np.ndarray, K0: np.ndarray, K1: np.ndarray, R_c1c0: np.ndarray, p_c1c0: np.ndarray):
    """Paper Sec. III-D: [S Ad_{T_C1C0} G0(K0)  -S G1(K1)] [g0dot; g1dot] = S V_L0L1,ref, the geodesic velocities of
    the contact frames on the object (g0dot, in {C0}'s y and z) and on the finger (g1dot, in {C1}'s) for the reference
    relative twist V_ref (in {C1}'s axes). Least squares when singular (two flat surfaces)."""
    Sx = np.zeros((4, 6))
    for i, r in enumerate(S_ROWS):
        Sx[i, r] = 1.0
    M = Sx @ np.hstack([adjoint(R_c1c0, p_c1c0) @ surface_jacobian(K0), -surface_jacobian(K1)])
    g = np.linalg.lstsq(M, Sx @ np.asarray(V_ref, float), rcond=1e-9)[0]
    return g[:2], g[2:]


def tangent_K(S3: np.ndarray, R: np.ndarray) -> np.ndarray:
    """2 x 2 shape operator in the (y, z) basis of the frame R (columns x, y, z) from the 3 x 3 shape operator."""
    B = R[:, 1:]
    return B.T @ S3 @ B


@dataclass
class FilterParams:
    k_min: float = 5.0           # 1/s, gain at rest (smoothing)
    k_max: float = 100.0         # 1/s, gain when the reference geodesic speed is large (tracking)
    alpha: float = 0.02          # m/s, geodesic speed above which the gain passes 95 % of its range
    source: str = "reference"    # "reference": S V_L0L1,ref (paper); "planned": the twist Eq. (2) plans


class ContactFilter:
    """The complementary filter of Eqs. (4)-(6) for one finger: frames {C0} on the object and {C1} on the finger move
    on the surfaces along the geodesic velocities of the roll-slide kinematics, corrected toward the measured GCF
    origins with a gain that is small when the reference geodesic speed is small. {C1}'s origin and normal are the
    impedance controller's action point and direction."""

    def __init__(self, finger: FingerSpec, obj: ObjectSpec, prm: FilterParams | None = None):
        self.fs, self.os = finger.surface, obj.surface
        self.prm = prm or FilterParams()
        self.c0 = self.c1 = None                  # body-frame points on the object and on the finger
        self.k = (0.0, 0.0)

    def reset(self):
        self.c0 = self.c1 = None

    def frames(self, mirror: Mirror, f: str):
        o, Ro = mirror.obj_pose()
        pb, Rb = mirror.body_pose(f)
        _, q0, n0 = self.os.closest(self.c0)
        _, q1, n1 = self.fs.closest(self.c1)
        axis = mirror.joint_axis(f)
        R_c0 = frame_from_normal(Ro @ n0, axis, np.array([0.0, 0.0, 1.0]))
        R_c1 = frame_from_normal(Rb @ n1, axis, np.array([0.0, 0.0, 1.0]))
        return (o + Ro @ q0, R_c0, q0, n0), (pb + Rb @ q1, R_c1, q1, n1)

    def update(self, mirror: Mirror, f: str, g: Gcf, V_ref: np.ndarray, dt: float):
        o, Ro = mirror.obj_pose()
        pb, Rb = mirror.body_pose(f)
        l0, l1 = Ro.T @ (g.p0 - o), Rb.T @ (g.p1 - pb)
        if self.c0 is None:
            self.c0, self.c1 = self.os.closest(l0)[1], self.fs.closest(l1)[1]
        (w0, R_c0, q0, n0), (w1, R_c1, q1, n1) = self.frames(mirror, f)
        K0 = tangent_K(Ro @ self.os.shape_operator(q0, n0) @ Ro.T, R_c0)
        K1 = tangent_K(Rb @ self.fs.shape_operator(q1, n1) @ Rb.T, R_c1)
        g0r, g1r = roll_slide_rates(V_ref, K0, K1, R_c1.T @ R_c0, R_c1.T @ (w0 - w1))
        p = self.prm
        k0 = p.k_min + (p.k_max - p.k_min) * (1.0 - math.exp(-3.0 * np.linalg.norm(g0r) / p.alpha))     # Eq. (6)
        k1 = p.k_min + (p.k_max - p.k_min) * (1.0 - math.exp(-3.0 * np.linalg.norm(g1r) / p.alpha))
        e0 = R_c0[:, 1:].T @ (Ro @ (l0 - q0))                                                            # E (p_L - p_C)
        e1 = R_c1[:, 1:].T @ (Rb @ (l1 - q1))
        gd0, gd1 = g0r + k0 * e0, g1r + k1 * e1                                                          # Eq. (5)
        self.c0 = self.os.geodesic_step(q0, Ro.T @ (R_c0[:, 1:] @ gd0) * dt)                             # Eq. (4)
        self.c1 = self.fs.geodesic_step(q1, Rb.T @ (R_c1[:, 1:] @ gd1) * dt)
        self.k = (k0, k1)
        (_, _, _, _), (w1n, R_c1n, _, _) = self.frames(mirror, f)
        return w1n, R_c1n[:, 0]


# ------------------------------------------------------------------------------------------------ commands

@dataclass
class FingerCommand:
    """One finger's sliders in the paper's teleoperation interface (Sec. III-E)."""
    mode: str = "free"                                   # "free", "pinch" or "pusher"
    v_ref: np.ndarray = field(default_factory=lambda: np.zeros(6))      # (spin, roll_y, roll_z, sep, slide_y, slide_z)
    weights: np.ndarray = field(default_factory=lambda: np.ones(6))     # w_i of Eq. (2), per component
    force: float = 0.0                                   # N: the applied force (the impedance's reference)
    follow_object: bool = True                           # use the object's reference twist in Eq. (1)


@dataclass
class HandCommand:
    fingers: dict[str, FingerCommand]
    object_twist: np.ndarray = field(default_factory=lambda: np.zeros(6))   # V_PB0 reference, object body frame
    s: float = 0.0                                       # Eq. (3) reduced-order speed about the pinch axis, rad/s
    pinch: tuple[str, str] | None = None                 # the fingers of the pinch that defines {M}
    pusher_twist: np.ndarray | None = None               # explicit object body twist for pusher fingers; replaces
                                                         # Eq. (3) (a pinch used as a prismatic joint: a translation)


@dataclass
class HomParams:
    dt: float = 0.01                     # s, controller period (paper: 500 Hz)
    w_damp: float = 1e-3
    w_healthy: float = 1e-5
    k_healthy: float = 1.0               # 1/s
    u_max: float = 2.0                   # rad/s
    aw_dist: float = 0.5e-3              # m: v_sep,ref > 0 is clamped to zero inside this distance (anti-windup)
    lead: float = 0.3                    # rad: |theta_ref - theta| <= lead
    gravity_ff: bool = True              # add tau_g / k_p to the servo targets
    use_filter: bool = True              # impedance action point from the complementary filter
    contact_force_min: float = 0.02      # N: a finger with less pad force counts as apart
    force_loop_gain: float = 0.0         # m/s per N: optional v_sep correction from the measured force (not in the paper)
    dls: float = 0.002                   # m/rad: damping of the gap-closing step of the servo impedance
    ell: float = ELL                     # m: linear rows of Eq. (2) are divided by ell (1.0: SI units, equal weights)
    filter: FilterParams = field(default_factory=FilterParams)


@dataclass
class Measurement:
    q: dict[str, np.ndarray]                             # joint angles per finger (servo bus / plant)
    obj_pos: np.ndarray                                  # object frame origin, world (tag tracker / plant)
    obj_quat: np.ndarray                                 # (w, x, y, z)
    cop: dict[str, tuple[np.ndarray, float]] = field(default_factory=dict)   # per finger: centre of pressure, |F_n|


class HomController:
    """The planner of Sec. III-C with the estimator of Sec. III-D, writing servo targets."""

    def __init__(self, mirror: Mirror, prm: HomParams | None = None, healthy: dict[str, np.ndarray] | None = None):
        self.mr = mirror
        self.prm = prm or HomParams()
        self.order = mirror.order
        self.theta_ref = None
        self.healthy = healthy
        self.filters = {f: ContactFilter(mirror.fingers[f], mirror.obj, self.prm.filter)
                        if mirror.fingers[f].surface is not None and mirror.obj.surface is not None else None
                        for f in self.order}
        self.M = None                                    # pinch frame {M} in the object frame (R, p)
        self.M_key = None

    def reset(self, theta0: dict[str, np.ndarray]):
        self.theta_ref = {f: np.asarray(theta0[f], float).copy() for f in self.order}
        if self.healthy is None:
            self.healthy = {f: v.copy() for f, v in self.theta_ref.items()}
        for flt in self.filters.values():
            if flt is not None:
                flt.reset()
        self.M = self.M_key = None

    def reanchor(self, f: str, theta: np.ndarray):
        """Set finger f's joint reference to its measured angles: the switch from motion to force control of a finger
        whose planned motion the object did not follow (a pusher stopped by a wall), so the reference wound up ahead of
        the finger does not stay in the servo spring as force."""
        if self.theta_ref is not None:
            self.theta_ref[f] = np.asarray(theta, float).copy()

    # Eq. (1)
    def eq1_rows(self, f: str, g: Gcf, V_obj_body: np.ndarray):
        """A (6 x n_f), b (6): the relative twist of {L1} with respect to the object, in {L1}'s axes, is A u + b for
        the finger's joint rates u, with the object moving at the reference body twist V_obj_body."""
        o, Ro = self.mr.obj_pose()
        Jp, Jr = self.mr.jac(f, g.p1)
        R1 = g.R1
        w_o = Ro @ V_obj_body[:3]
        v_p1 = Ro @ V_obj_body[3:] + np.cross(w_o, g.p1 - o)          # object's material velocity at {L1}'s origin
        A = np.vstack([R1.T @ Jr, R1.T @ Jp])
        b = -np.concatenate([R1.T @ w_o, R1.T @ v_p1])
        return A, b

    def gcf(self, f: str, meas: Measurement) -> Gcf:
        cop = meas.cop.get(f)
        if (cop is not None and cop[1] >= self.prm.contact_force_min and self.mr.fingers[f].surface is not None
                and self.mr.obj.surface is not None):
            return self.mr.gcf_from_cop(f, np.asarray(cop[0], float))
        return self.mr.gcf_witness(f)

    def step(self, meas: Measurement, cmd: HandCommand):
        """One control period: GCFs, Eq. (2) per finger, filter, servo targets. Returns (q_cmd per finger, info)."""
        prm, mr = self.prm, self.mr
        dt = prm.dt
        if self.theta_ref is None:
            self.reset(meas.q)
        mr.set_state(meas.q, meas.obj_pos, meas.obj_quat)
        o, Ro = mr.obj_pose()
        gcfs = {f: self.gcf(f, meas) for f in self.order}
        if cmd.pinch is not None and abs(cmd.s) > 0 and self.M_key != cmd.pinch:
            a, b_ = cmd.pinch                            # {M} is fixed to the object when the pusher starts
            self.M = pinch_frame(gcfs[a].p0, gcfs[b_].p0, o, Ro)
            self.M_key = cmd.pinch
        if cmd.pinch is None:
            self.M = self.M_key = None
        out, info = {}, {}
        for f in self.order:
            c = cmd.fingers.get(f, FingerCommand())
            g = gcfs[f]
            V = np.asarray(cmd.object_twist, float) if c.follow_object else np.zeros(6)
            if c.mode == "pusher" and cmd.pusher_twist is not None:
                V = V + np.asarray(cmd.pusher_twist, float)
            elif c.mode == "pusher" and self.M is not None:
                V = V + pusher_body_twist(cmd.s, *self.M)
            v_ref = np.asarray(c.v_ref, float).copy()
            Fm = meas.cop.get(f, (None, 0.0))[1] if meas.cop else 0.0
            touching = Fm >= prm.contact_force_min or g.dist <= 0.0
            if prm.force_loop_gain > 0 and touching and c.force > 0:
                v_ref[SEP] += prm.force_loop_gain * (c.force - Fm)
            if g.dist < prm.aw_dist and v_ref[SEP] > 0 and not (prm.force_loop_gain > 0 and touching):
                v_ref[SEP] = 0.0                          # anti-windup (Sec. III-C)
            A, b = self.eq1_rows(f, g, V)
            th = np.asarray(meas.q[f], float)
            u = solve_eq2(A, b, v_ref, c.weights, th, self.theta_ref[f], mr.lo[f], mr.hi[f], dt,
                          self.healthy.get(f) if self.healthy else None, prm.w_damp, prm.w_healthy, prm.k_healthy,
                          prm.u_max, scale=np.array([1.0, 1.0, 1.0] + [1.0 / prm.ell] * 3))
            tr = self.theta_ref[f] + dt * u
            tr = np.clip(tr, th - prm.lead, th + prm.lead)
            self.theta_ref[f] = np.clip(tr, mr.lo[f], mr.hi[f])
            # the impedance: action point and direction from the filter ({C1}) or the GCF ({L1})
            p_act, n_act = g.p1, g.x1
            flt = self.filters[f]
            if flt is not None and prm.use_filter:
                V_f = (A @ u + b) if prm.filter.source == "planned" else v_ref
                p_act, n_act = flt.update(mr, f, g, V_f, dt)
            offset = np.zeros(len(th))
            kp = mr.kp[f]
            if c.force > 0:
                # the impedance's reference penetration is measured from the object's surface: move the action point
                # across the remaining gap (damped least squares), then press with F through the servo spring
                Jp, _ = mr.jac(f, p_act)
                gap = max(g.dist, 0.0)
                if gap > 0:
                    offset += Jp.T @ np.linalg.solve(Jp @ Jp.T + prm.dls ** 2 * np.eye(3), gap * n_act)
                offset += Jp.T @ (c.force * n_act) / kp
            if prm.gravity_ff:
                offset += mr.gravity(f) / kp
            out[f] = np.clip(self.theta_ref[f] + offset, mr.lo[f], mr.hi[f])
            info[f] = {"dist": g.dist, "source": g.source, "u": u, "v_plan": A @ u + b, "v_ref": v_ref,
                       "touching": touching, "F": Fm, "p_act": p_act, "n_act": n_act,
                       "k_filter": flt.k if flt is not None else None}
        return out, info
