"""Reference trajectory loader for Opt2Skill-style warm-start RL.

Source: `results/phase1/<tag>/foundational/<task>/<run_id>/best_rollout.npz`
plus `summary.json`, written by `scripts/phase1_optimize_grasp.py` via
`Phase1GraspEvaluator._save_rollout` (see `phase1_common.py`).

The on-disk npz holds the *full state* trajectory under a fixed finger
control vector — exactly the reference we want a learned policy to imitate.
We slice palm pose / object pose out of `qpos` rather than re-extending the
npz writer; the canonical column layout for a *frozen* (rigid-morphology)
cube scene is:

    qpos[:,  0:3]   object position
    qpos[:,  3:7]   object quaternion (wxyz)
    qpos[:,  7:13]  palm pose (palm_px, palm_py, palm_pz, palm_rx, palm_ry, palm_rz)
    qpos[:, 13:22]  9 finger joints (thumb_yaw, thumb_mcp, thumb_pip,
                                     index_yaw, index_mcp, index_pip,
                                     middle_yaw, middle_mcp, middle_pip)

We do NOT hardcode this order — column indices are resolved at load time
from a passed-in mujoco model (jnt_qposadr) so the loader stays robust if
the scene XML is reordered.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


# Canonical finger joint names — matches `phase1_common.py` actuator ordering.
FINGER_JOINT_NAMES: tuple[str, ...] = (
    "thumb_yaw", "thumb_mcp", "thumb_pip",
    "index_yaw", "index_mcp", "index_pip",
    "middle_yaw", "middle_mcp", "middle_pip",
)
PALM_JOINT_NAMES: tuple[str, ...] = (
    "palm_px", "palm_py", "palm_pz", "palm_rx", "palm_ry", "palm_rz",
)


@dataclass
class QposLayout:
    """qpos column indices, resolved against a mujoco model at load time."""
    object_pos: slice          # 3
    object_quat: slice         # 4 (wxyz)
    palm: tuple[int, ...]      # 6 — one index per PALM_JOINT_NAMES entry
    fingers: tuple[int, ...]   # 9 — one index per FINGER_JOINT_NAMES entry

    @classmethod
    def from_model(cls, model, object_freejoint_name: str = "") -> "QposLayout":
        """Build a layout from a `mujoco.MjModel`.

        The object freejoint is assumed to occupy the first 7 qpos entries
        (positions 0..6 = object pos+quat). Subsequent palm/finger joints
        are looked up by name. Pass an explicit `object_freejoint_name` if
        the scene has multiple free bodies.
        """
        import mujoco

        def jid(name: str) -> int:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"joint '{name}' not in model")
            return int(model.jnt_qposadr[jid])

        palm_ids = tuple(jid(n) for n in PALM_JOINT_NAMES)
        finger_ids = tuple(jid(n) for n in FINGER_JOINT_NAMES)
        return cls(
            object_pos=slice(0, 3),
            object_quat=slice(3, 7),
            palm=palm_ids,
            fingers=finger_ids,
        )


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """SLERP between two unit quaternions (wxyz), t in [0,1]."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / (np.linalg.norm(out) + 1e-12)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s0 = np.sin((1.0 - t) * theta) / np.sin(theta)
    s1 = np.sin(t * theta) / np.sin(theta)
    return s0 * q0 + s1 * q1


@dataclass
class ReferenceTrajectory:
    """In-memory reference rollout, sample-able at arbitrary continuous t.

    Use `from_run_dir` to construct; do not instantiate directly.
    """
    qpos: np.ndarray             # (T, nq)
    qvel: np.ndarray             # (T, nv)
    contacts: np.ndarray         # (T,)
    cube_z: np.ndarray           # (T,)
    finger_ctrl: np.ndarray      # (9,) — constant across this rollout
    layout: QposLayout
    dt: float                    # sim step (e.g. 0.002 s)
    settle_steps: int
    lift_steps: int
    hold_steps: int
    pivot_steps: int
    best_score: float

    @property
    def n_steps(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def duration(self) -> float:
        return self.n_steps * self.dt

    @classmethod
    def from_run_dir(
        cls,
        run_dir: Path | str,
        model,
        dt: float = 0.002,
    ) -> "ReferenceTrajectory":
        run_dir = Path(run_dir)
        npz = np.load(run_dir / "best_rollout.npz")
        with (run_dir / "summary.json").open() as f:
            summary = json.load(f)
        layout = QposLayout.from_model(model)
        qpos = np.asarray(npz["qpos"], dtype=np.float64)
        if qpos.shape[1] != model.nq:
            raise ValueError(
                f"reference qpos has {qpos.shape[1]} cols, model.nq={model.nq}; "
                "the reference run must use the same frozen scene as the env."
            )
        eval_cfg = summary.get("eval_config", {}) or {}
        return cls(
            qpos=qpos,
            qvel=np.asarray(npz["qvel"], dtype=np.float64),
            contacts=np.asarray(npz["contacts"], dtype=np.float64),
            cube_z=np.asarray(npz["cube_z"], dtype=np.float64),
            finger_ctrl=np.asarray(npz["best_finger_ctrl"], dtype=np.float64).reshape(-1),
            layout=layout,
            dt=dt,
            settle_steps=int(eval_cfg.get("settle_steps", 240)),
            lift_steps=int(eval_cfg.get("lift_steps", 220)),
            hold_steps=int(eval_cfg.get("hold_steps", 140)),
            pivot_steps=int(eval_cfg.get("pivot_steps", 0)),
            best_score=float(summary.get("best_score", float("nan"))),
        )

    # ------- indexed sample-at-time helpers -----------------------------

    def _frac_index(self, t: float) -> tuple[int, int, float]:
        """Convert continuous time `t` (seconds) into (lo, hi, alpha).

        Out-of-range `t` clamps to the trajectory endpoints — this is the
        desired behaviour for RL lookahead at the tail of an episode.
        """
        x = t / self.dt
        x = max(0.0, min(x, self.n_steps - 1))
        lo = int(np.floor(x))
        hi = min(lo + 1, self.n_steps - 1)
        alpha = x - lo
        return lo, hi, alpha

    def _lerp_row(self, arr: np.ndarray, t: float) -> np.ndarray:
        lo, hi, a = self._frac_index(t)
        return (1.0 - a) * arr[lo] + a * arr[hi]

    def object_pos_at(self, t: float) -> np.ndarray:
        return self._lerp_row(self.qpos[:, self.layout.object_pos], t)

    def object_quat_at(self, t: float) -> np.ndarray:
        lo, hi, a = self._frac_index(t)
        q0 = self.qpos[lo, self.layout.object_quat]
        q1 = self.qpos[hi, self.layout.object_quat]
        return _slerp(q0, q1, a)

    def object_pose_at(self, t: float) -> np.ndarray:
        return np.concatenate([self.object_pos_at(t), self.object_quat_at(t)])

    def palm_qpos_at(self, t: float) -> np.ndarray:
        idx = list(self.layout.palm)
        return self._lerp_row(self.qpos[:, idx], t)

    def finger_qpos_at(self, t: float) -> np.ndarray:
        idx = list(self.layout.fingers)
        return self._lerp_row(self.qpos[:, idx], t)

    def finger_ctrl_at(self, t: float) -> np.ndarray:
        # CEM emitted a single (9,) control vector held for the entire
        # rollout. If trajectory-CEM ever produces per-segment ctrls
        # they'd be a (K, 9) array; switch on shape and interpolate by phase.
        return self.finger_ctrl.copy()

    def contact_count_at(self, t: float) -> float:
        lo, hi, a = self._frac_index(t)
        return float((1.0 - a) * self.contacts[lo] + a * self.contacts[hi])

    def cube_z_at(self, t: float) -> float:
        lo, hi, a = self._frac_index(t)
        return float((1.0 - a) * self.cube_z[lo] + a * self.cube_z[hi])

    # ------- vectorised lookahead (batches of times) --------------------

    def batch_at(self, ts: Sequence[float]) -> dict[str, np.ndarray]:
        """Vectorised: stack per-time references into arrays for obs lookahead."""
        ts = np.asarray(ts, dtype=np.float64)
        out: dict[str, list[np.ndarray]] = {
            "object_pos": [], "object_quat": [],
            "palm_qpos": [], "finger_qpos": [], "finger_ctrl": [],
        }
        for t in ts:
            out["object_pos"].append(self.object_pos_at(float(t)))
            out["object_quat"].append(self.object_quat_at(float(t)))
            out["palm_qpos"].append(self.palm_qpos_at(float(t)))
            out["finger_qpos"].append(self.finger_qpos_at(float(t)))
            out["finger_ctrl"].append(self.finger_ctrl_at(float(t)))
        return {k: np.stack(v) for k, v in out.items()}
