"""Differential IK for an arm-mounted hand: a palm pose in, six joint angles out.

The chain controller speaks one language, a WORLD PALM POSE (R, p). On the floating-palm scene
that pose is six actuator set-points and the conversion is algebra. On an arm it is an IK
problem with limits, a reach envelope and singularities, and it can simply fail -- which is the
whole reason for running the same controller on both.

mink does the solving. It is pinned to 1.1 in the `arm` extra because 1.2 pulls mujoco 3.12,
and every physics number in this repo was measured on 3.6.

The solver runs on a SEPARATE arm-only model. mink's `Configuration` integrates over every DOF
the model has, so handing it the task scene would let it satisfy a palm-pose target by bending
the fingers or by teleporting the screwdriver's free joint.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

UR5E_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


class ArmIK:
    """Solve a site pose on an arm-only model; report the residual honestly."""

    # daqp, not quadprog: qpsolvers ships daqp with mink's dependency set and quadprog is
    # an optional extra that is not installed here. The choice is a QP backend, not a method.
    def __init__(self, ik_xml: Path, site: str = "palm_site",
                 joints: tuple = UR5E_JOINTS, solver: str = "daqp"):
        self.model = mujoco.MjModel.from_xml_path(str(ik_xml))
        self.data = mujoco.MjData(self.model)
        self.site = site
        self.joints = joints
        self.solver = solver
        self.adr = [self.model.jnt_qposadr[self.model.joint(j).id] for j in joints]
        import mink
        self._mink = mink

    def fk(self, q: np.ndarray):
        """Where the palm site lands for a joint vector: (R, p)."""
        self.data.qpos[:] = 0.0
        for a, v in zip(self.adr, q):
            self.data.qpos[a] = float(v)
        mujoco.mj_forward(self.model, self.data)
        s = self.data.site(self.site)
        return s.xmat.reshape(3, 3).copy(), s.xpos.copy()

    def solve(self, R: np.ndarray, p: np.ndarray, q_seed: np.ndarray,
              iters: int = 400, dt: float = 0.02, posture_cost: float = 1e-3):
        """Joint vector putting the palm site at (R, p), plus (pos_err_m, rot_err_rad).

        Seeded from `q_seed` and regularised toward it, so a sequence of nearby targets returns
        a continuous joint path instead of hopping between IK branches -- which on a 6R arm is a
        wrist flip, i.e. the hand rotating 180 degrees about the tool axis between two set-points
        1 mm apart.
        """
        mink = self._mink
        cfg = mink.Configuration(self.model)
        q0 = np.zeros(self.model.nq)
        for a, v in zip(self.adr, q_seed):
            q0[a] = float(v)
        cfg.update(q0)
        task = mink.FrameTask(frame_name=self.site, frame_type="site",
                              position_cost=1.0, orientation_cost=1.0, lm_damping=1e-2)
        task.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(np.asarray(R, float)), np.asarray(p, float)))
        posture = mink.PostureTask(self.model, cost=posture_cost)
        posture.set_target(q0)
        limits = [mink.ConfigurationLimit(self.model)]
        for _ in range(iters):
            v = mink.solve_ik(cfg, [task, posture], dt, self.solver,
                              damping=1e-3, limits=limits)
            cfg.integrate_inplace(v, dt)
            if float(np.linalg.norm(task.compute_error(cfg))) < 1e-6:
                break
        q = np.array([cfg.q[a] for a in self.adr])
        Rf, pf = self.fk(q)
        rot = float(np.arccos(np.clip((np.trace(Rf.T @ np.asarray(R, float)) - 1) / 2, -1, 1)))
        return q, float(np.linalg.norm(pf - np.asarray(p, float))), rot
