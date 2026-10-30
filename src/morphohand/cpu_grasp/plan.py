"""Open-loop grasp plan (P0) for a cylinder of known radius: pre-shape -> close -> lift -> hold.

The RL policy is a bounded residual on this plan (as in the manta-hand proposal and the GPU
stack's LerpFinger + residual), so the plan defines the task's starting competence and the
residual only has to fix what the plan gets wrong for this size / mass.

Keyframes come from `kin.grasp_ik` on the D1 hand (table grasp, 2026-10-30): the UR5 sets the palm
height, the fingertips stay FLOOR_CLEAR off the table, PIP joints stay >= 0 (the real index PIP
stops at -2.9 deg; hyperextending them was why the first plan's PIPs bent backwards), and the IK is
seeded from the D1 bench grip so the grasp sits where the bench turn (judge.py) was designed:
    q_pre    pads backed off the shaft by `standoff` along their normals (same palm height)
    q_grip   pads `squeeze` inside the shaft -- with a 1.5 N*m/rad servo that tracking error is
             the grip force (F ~ kp * dq / lever), so squeeze is the force knob
Timeline (seconds; the env runs the policy at 50 Hz):
    [0, t_close)              hold q_pre
    [t_close, t_grip)         lerp q_pre -> q_grip
    [t_grip, t_lift)          hold q_grip (let the squeeze build)
    [t_lift, t_lift + T_LIFT) palm rises by LIFT_DZ (scripted arm, never in the action space)
    then                      hold
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import kin, scene

T_CLOSE, T_GRIP, T_LIFT = 0.2, 0.7, 1.0
T_LIFT_DUR = 0.8
LIFT_DZ = 0.15     # 2026-10-30: was 0.08; standing a 100 mm cylinder up below the grip hit the table
T_GRASP_END = T_LIFT + T_LIFT_DUR + 0.4        # 2.4 s: the grasp episode; the judge takes over here


@dataclass
class GraspPlan:
    radius: float
    q_pre: np.ndarray
    q_grip: np.ndarray
    palm_z: float           # palm_z joint value at grasp height
    obj_pos: np.ndarray     # object centre at rest
    feasible: bool

    def q(self, t: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t, float)
        a = np.clip((t - T_CLOSE) / (T_GRIP - T_CLOSE), 0, 1)
        a = a * a * (3 - 2 * a)                                  # smoothstep close
        return self.q_pre + (self.q_grip - self.q_pre) * a[..., None]

    def palm(self, t: np.ndarray | float) -> np.ndarray:
        t = np.asarray(t, float)
        a = np.clip((t - T_LIFT) / T_LIFT_DUR, 0, 1)
        a = a * a * (3 - 2 * a)
        return self.palm_z + LIFT_DZ * a


SQUEEZE = 0.008        # 2026-10-30: 4 mm dropped 100-150 g cylinders once pad friction was realistic
FLOOR_CLEAR = 0.002    # fingertips never touch the table
# D1 bench grip (deploy/sv1_w6689_b050_poses.txt "open/grip"), deg -> rad
Q_D1 = np.radians([32.50, 39.58, 0.0, -23.15, 32.25, 18.87, -0.61, 24.92, 40.68])
import os as _os
W_FLOOR = float(_os.environ.get("PLAN_W_FLOOR", 100.0))   # 1e4 changed nothing (2026-10-30)
PRE_MODE = _os.environ.get("PLAN_PRE", "horizontal")   # "virtual" (larger-cylinder grasp) broke grasps: E1 0.02
PALM_UP = 0.01         # initial palm guess above touching height (the IK then frees palm z)


def make_plan(model, radius: float, squeeze: float = SQUEEZE, standoff: float = 0.012,
              K: kin.Kinematics | None = None) -> GraspPlan:
    K = K or kin.Kinematics(model)
    K.floor_clear = FLOOR_CLEAR
    K.lo[[2, 5, 8]] = np.maximum(K.lo[[2, 5, 8]], 0.0)           # PIP >= 0
    obj = scene.object_rest_pose(model, radius)[:3]
    palm_base = model.body("palm_pose").pos[2]
    yaw_z = model.body("thumb_mount").pos[2]
    p0 = (radius + 0.0505) - (palm_base + yaw_z) + PALM_UP          # 50.5 mm = D1 grip depth
    kw = dict(w_y=0.0, w_eq=0.1, w_floor=W_FLOOR)
    touch = K.grasp_ik(obj, radius, squeeze=0.0, q0=Q_D1, palm0=p0, **kw)
    grip = K.grasp_ik(obj, radius, squeeze=squeeze, q0=touch.q, palm0=touch.palm_z, palm_free=False, **kw)
    # pre-shape: the same grasp on a cylinder `standoff` larger (same centre), so the pads sit
    # clear of the shaft AND the table term still applies (backing off along the pad normal, the
    # first version, pushed the fingertips into the table: contact from step 0 on every size)
    if PRE_MODE == "virtual":
        pre = K.grasp_ik(obj, radius + standoff, squeeze=0.0, q0=touch.q, palm0=touch.palm_z, palm_free=False, **kw)
    else:
        face, nrm = K._fk(np.concatenate([touch.q, [touch.palm_z]]))
        nh = nrm.copy(); nh[:, 2] = 0.0
        nh /= np.linalg.norm(nh, axis=1, keepdims=True) + 1e-9
        pre = K.track_ik(face - nh * standoff + np.array([0, 0, 0.003]), nrm, touch.q, touch.palm_z)
    feasible = touch.residual < 2e-3 and touch.floor_z > 0
    return GraspPlan(radius, pre.q, grip.q, touch.palm_z, obj, feasible)


@lru_cache(maxsize=None)
def plan_table(radii: tuple, squeeze: float = SQUEEZE) -> dict:
    """Plans for a grid of radii on the default D1 model (cached; the env interpolates)."""
    m = scene.build()
    K = kin.Kinematics(m)
    return {r: make_plan(m, r, squeeze=squeeze, K=K) for r in radii}
