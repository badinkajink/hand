"""One controller, two wrists: the floating gantry and a UR5e, behind the same interface.

Everything the chain asks of the palm is a WORLD POSE (R, p) -- the lift, the rigid re-pose
transfer, the press, the re-index translation. On the floating-palm scene that is six actuators
and a bit of algebra; on the arm it is differential IK against joint limits and a reach envelope.
Putting both behind `read / write / solve / cmd_pose` means the chain probe contains no `if arm:`
anywhere, so a difference between the two runs is the wrist's, not the script's.

Two rules both drivers obey, and they are the ones that make the comparison mean anything:

  * `cmd_pose` is the pose of the COMMAND, never of the achieved body. Under load the plant lags,
    and a controller that re-references its own deflection washes out its own preload -- the
    same trap that bled this hand's grip from 18.4 N to 0.25 N over seven reissues of a
    "constant" grip command.
  * `solve` is seeded from the current command and regularised toward it, so a sequence of
    nearby targets returns a continuous joint path. On a 6R arm the alternative is a branch hop:
    the wrist flipping 180 degrees between two set-points a millimetre apart.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

GANTRY_JOINTS = ("palm_px", "palm_py", "palm_pz", "palm_rx", "palm_ry", "palm_rz")


def _euler_xyz(m, R: np.ndarray, p: np.ndarray, body: str = "palm_pose"):
    """Gantry joint values putting `body` at world (R, p).

    MuJoCo composes a body's joints in declaration order, so with px,py,pz before rx,ry,rz the
    slides are world-axis translations and the hinges are an INTRINSIC X-Y-Z Euler triple,
    R = Rx(a) Ry(b) Rz(c). Getting that order wrong is silent: the palm lands somewhere
    plausible and the downstream ring IK reports a residual nobody can explain.
    """
    assert np.allclose(m.body(body).quat, [1, 0, 0, 0], atol=1e-9), \
        f"{body} carries a body quat; this Euler decomposition assumes identity"
    b = float(np.arcsin(np.clip(R[0, 2], -1.0, 1.0)))
    a = float(np.arctan2(-R[1, 2], R[2, 2]))
    c = float(np.arctan2(-R[0, 1], R[0, 0]))
    t = np.asarray(p, float) - np.asarray(m.body(body).pos, float)
    return np.array([t[0], t[1], t[2], a, b, c])


class GantryPalm:
    """The Harry Potter hand: six position actuators on the palm body itself."""

    kind = "gantry"

    def __init__(self, m, d):
        self.m, self.d = m, d
        self.joints = GANTRY_JOINTS
        self.acts = [next(k for k in range(m.nu) if m.actuator(k).name == f"a_{j}")
                     for j in GANTRY_JOINTS]

    def read(self) -> np.ndarray:
        return np.array([float(self.d.ctrl[a]) for a in self.acts])

    def write(self, u: np.ndarray) -> None:
        for a, v in zip(self.acts, u):
            self.d.ctrl[a] = float(v)

    def joint_dict(self, u: np.ndarray) -> dict:
        return {j: float(v) for j, v in zip(self.joints, u)}

    def fk(self, u: np.ndarray):
        """Palm world pose for a command vector. Closed form: the gantry IS the pose."""
        R = np.eye(3)
        for ang, ax in ((u[3], 0), (u[4], 1), (u[5], 2)):
            c, s = np.cos(ang), np.sin(ang)
            Rm = {0: np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
                  1: np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
                  2: np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[ax]
            R = R @ Rm
        return R, np.asarray(self.m.body("palm_pose").pos, float) + u[:3]

    def cmd_pose(self):
        return self.fk(self.read())

    def solve(self, R: np.ndarray, p: np.ndarray, seed: np.ndarray | None = None):
        return _euler_xyz(self.m, R, p), 0.0, 0.0


class ArmPalm:
    """A UR5e carrying the same palm, driven by mink differential IK."""

    kind = "arm"

    def __init__(self, m, d, ik_xml: Path):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from morphohand.tools.arm_ik import ArmIK, UR5E_JOINTS
        self.m, self.d = m, d
        self.joints = UR5E_JOINTS
        self.ik = ArmIK(ik_xml)
        self.acts = [next(k for k in range(m.nu) if m.actuator(k).name == n)
                     for n in ("shoulder_pan", "shoulder_lift", "elbow",
                               "wrist_1", "wrist_2", "wrist_3")]
        self.worst_pos = 0.0
        self.worst_rot = 0.0
        self.fails = 0

    def read(self) -> np.ndarray:
        return np.array([float(self.d.ctrl[a]) for a in self.acts])

    def write(self, u: np.ndarray) -> None:
        for a, v in zip(self.acts, u):
            self.d.ctrl[a] = float(v)

    def joint_dict(self, u: np.ndarray) -> dict:
        return {j: float(v) for j, v in zip(self.joints, u)}

    def fk(self, u: np.ndarray):
        return self.ik.fk(np.asarray(u, float))

    def cmd_pose(self):
        return self.fk(self.read())

    def solve(self, R: np.ndarray, p: np.ndarray, seed: np.ndarray | None = None):
        q, ep, er = self.ik.solve(R, p, self.read() if seed is None else seed)
        self.worst_pos = max(self.worst_pos, ep)
        self.worst_rot = max(self.worst_rot, er)
        # 1 mm / 0.5 deg: below the hand's own IK residuals, so a pose the arm cannot make is
        # counted here and not blamed on the fingers downstream.
        if ep > 1e-3 or er > np.radians(0.5):
            self.fails += 1
        return q, ep, er


def make(m, d, ik_xml: Path | None):
    return GantryPalm(m, d) if ik_xml is None else ArmPalm(m, d, ik_xml)
