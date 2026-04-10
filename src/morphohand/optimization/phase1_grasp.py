from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np


@dataclass
class Phase1EvalConfig:
    settle_steps: int = 240
    lift_steps: int = 220
    hold_steps: int = 140
    lift_delta_z: float = 0.05
    objective_weight_distance: float = 2.0
    objective_weight_contact: float = 0.4
    objective_weight_lift: float = 35.0
    objective_weight_velocity_penalty: float = 0.15


@dataclass
class Phase1OptimizationConfig:
    iterations: int = 24
    population: int = 40
    elite_fraction: float = 0.2
    sigma_init: float = 0.20
    seed: int = 0


class Phase1GraspEvaluator:
    """Scores finger control vectors for a fixed morphology scene."""

    def __init__(self, scene_xml: Path, keyframe: str = "open", cfg: Phase1EvalConfig | None = None) -> None:
        self.scene_xml = Path(scene_xml)
        self.cfg = cfg or Phase1EvalConfig()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)

        self.keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if self.keyframe_id < 0:
            raise ValueError(f"Keyframe '{keyframe}' not found in {self.scene_xml}")

        self.pose_actuator_names = [
            "a_palm_px",
            "a_palm_py",
            "a_palm_pz",
            "a_palm_rx",
            "a_palm_ry",
            "a_palm_rz",
        ]
        self.finger_actuator_names = [
            "a_thumb_yaw",
            "a_thumb_mcp",
            "a_thumb_pip",
            "a_index_yaw",
            "a_index_mcp",
            "a_index_pip",
            "a_middle_yaw",
            "a_middle_mcp",
            "a_middle_pip",
        ]

        self.pose_actuator_ids = self._actuator_ids(self.pose_actuator_names)
        self.finger_actuator_ids = self._actuator_ids(self.finger_actuator_names)

        self.cube_body_id = self._require_body("cube")
        self.tip_body_ids = [
            self._require_body("thumb_tip"),
            self._require_body("index_tip"),
            self._require_body("middle_tip"),
        ]

        self._reset_to_keyframe()
        self.initial_ctrl = self.data.ctrl.copy()
        self.initial_palm_pz_target = float(self.initial_ctrl[self.pose_actuator_ids[2]])

        self.finger_ctrl_min, self.finger_ctrl_max = self._finger_ctrl_bounds()
        self.cube_half_extents = np.array([0.02, 0.02, 0.02], dtype=np.float64)

    def _actuator_ids(self, names: list[str]) -> np.ndarray:
        ids = []
        for name in names:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if idx < 0:
                raise ValueError(f"Actuator '{name}' not found in scene")
            ids.append(idx)
        return np.array(ids, dtype=np.int32)

    def _require_body(self, name: str) -> int:
        idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if idx < 0:
            raise ValueError(f"Body '{name}' not found in scene")
        return int(idx)

    def _finger_ctrl_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        mins = []
        maxs = []
        for actuator_id in self.finger_actuator_ids:
            mins.append(self.model.actuator_ctrlrange[actuator_id, 0])
            maxs.append(self.model.actuator_ctrlrange[actuator_id, 1])
        return np.asarray(mins), np.asarray(maxs)

    def _reset_to_keyframe(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe_id)
        mujoco.mj_forward(self.model, self.data)

    def _build_full_ctrl(self, finger_ctrl: np.ndarray, lift: bool = False) -> np.ndarray:
        ctrl = self.initial_ctrl.copy()
        ctrl[self.finger_actuator_ids] = finger_ctrl
        if lift:
            ctrl[self.pose_actuator_ids[2]] = self.initial_palm_pz_target + self.cfg.lift_delta_z
        return ctrl

    def _step_with_ctrl(self, ctrl: np.ndarray, steps: int) -> None:
        for _ in range(steps):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

    def _cube_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.xpos[self.cube_body_id].copy()
        rot = self.data.xmat[self.cube_body_id].reshape(3, 3).copy()
        return pos, rot

    def _tip_positions(self) -> np.ndarray:
        return np.vstack([self.data.xpos[bid] for bid in self.tip_body_ids])

    def _tip_distances_to_cube(self) -> np.ndarray:
        cube_pos, cube_rot = self._cube_pose()
        tip_world = self._tip_positions()
        local = (tip_world - cube_pos) @ cube_rot
        clipped = np.clip(local, -self.cube_half_extents, self.cube_half_extents)
        closest_world = clipped @ cube_rot.T + cube_pos
        return np.linalg.norm(tip_world - closest_world, axis=1)

    def _cube_tip_contact_count(self) -> int:
        count = 0
        tip_bodies = set(self.tip_body_ids)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[contact.geom1])
            b2 = int(self.model.geom_bodyid[contact.geom2])
            if b1 == self.cube_body_id and b2 in tip_bodies:
                count += 1
            elif b2 == self.cube_body_id and b1 in tip_bodies:
                count += 1
        return count

    def evaluate(self, finger_ctrl: np.ndarray) -> tuple[float, dict[str, float]]:
        finger_ctrl = np.asarray(finger_ctrl, dtype=np.float64)
        finger_ctrl = np.clip(finger_ctrl, self.finger_ctrl_min, self.finger_ctrl_max)

        self._reset_to_keyframe()

        settle_ctrl = self._build_full_ctrl(finger_ctrl, lift=False)
        self._step_with_ctrl(settle_ctrl, self.cfg.settle_steps)

        distances = self._tip_distances_to_cube()
        contact_count = self._cube_tip_contact_count()
        z_before = float(self.data.xpos[self.cube_body_id, 2])

        lift_ctrl = self._build_full_ctrl(finger_ctrl, lift=True)
        peak_z = z_before
        for _ in range(self.cfg.lift_steps):
            self.data.ctrl[:] = lift_ctrl
            mujoco.mj_step(self.model, self.data)
            peak_z = max(peak_z, float(self.data.xpos[self.cube_body_id, 2]))

        self._step_with_ctrl(lift_ctrl, self.cfg.hold_steps)
        z_after_hold = float(self.data.xpos[self.cube_body_id, 2])
        cube_vel_norm = float(np.linalg.norm(self.data.qvel[:6]))

        lift_amount = peak_z - z_before
        mean_dist = float(np.mean(distances))

        score = (
            self.cfg.objective_weight_lift * lift_amount
            - self.cfg.objective_weight_distance * mean_dist
            + self.cfg.objective_weight_contact * float(contact_count)
            - self.cfg.objective_weight_velocity_penalty * cube_vel_norm
        )

        metrics = {
            "score": float(score),
            "mean_tip_distance": mean_dist,
            "tip_distance_thumb": float(distances[0]),
            "tip_distance_index": float(distances[1]),
            "tip_distance_middle": float(distances[2]),
            "cube_tip_contacts": float(contact_count),
            "cube_z_before_lift": z_before,
            "cube_z_peak": peak_z,
            "cube_z_after_hold": z_after_hold,
            "cube_lift": lift_amount,
            "cube_vel_norm": cube_vel_norm,
        }
        return float(score), metrics

    def rollout(self, finger_ctrl: np.ndarray) -> dict[str, np.ndarray]:
        finger_ctrl = np.asarray(finger_ctrl, dtype=np.float64)
        finger_ctrl = np.clip(finger_ctrl, self.finger_ctrl_min, self.finger_ctrl_max)
        self._reset_to_keyframe()

        settle_ctrl = self._build_full_ctrl(finger_ctrl, lift=False)
        lift_ctrl = self._build_full_ctrl(finger_ctrl, lift=True)

        total_steps = self.cfg.settle_steps + self.cfg.lift_steps + self.cfg.hold_steps
        qpos = np.zeros((total_steps, self.model.nq), dtype=np.float64)
        qvel = np.zeros((total_steps, self.model.nv), dtype=np.float64)
        cube_z = np.zeros(total_steps, dtype=np.float64)
        contacts = np.zeros(total_steps, dtype=np.float64)

        for t in range(total_steps):
            if t < self.cfg.settle_steps:
                self.data.ctrl[:] = settle_ctrl
            else:
                self.data.ctrl[:] = lift_ctrl
            mujoco.mj_step(self.model, self.data)
            qpos[t] = self.data.qpos
            qvel[t] = self.data.qvel
            cube_z[t] = self.data.xpos[self.cube_body_id, 2]
            contacts[t] = self._cube_tip_contact_count()

        return {
            "qpos": qpos,
            "qvel": qvel,
            "cube_z": cube_z,
            "contacts": contacts,
        }

    def render_rollout_gif(
        self,
        finger_ctrl: np.ndarray,
        output_gif: Path,
        width: int = 720,
        height: int = 540,
        fps: int = 25,
        frame_stride: int = 4,
    ) -> Path:
        finger_ctrl = np.asarray(finger_ctrl, dtype=np.float64)
        finger_ctrl = np.clip(finger_ctrl, self.finger_ctrl_min, self.finger_ctrl_max)
        self._reset_to_keyframe()

        settle_ctrl = self._build_full_ctrl(finger_ctrl, lift=False)
        lift_ctrl = self._build_full_ctrl(finger_ctrl, lift=True)

        max_width = int(self.model.vis.global_.offwidth)
        max_height = int(self.model.vis.global_.offheight)
        render_width = min(width, max_width)
        render_height = min(height, max_height)
        renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        total_steps = self.cfg.settle_steps + self.cfg.lift_steps + self.cfg.hold_steps
        frames: list[np.ndarray] = []

        for t in range(total_steps):
            if t < self.cfg.settle_steps:
                self.data.ctrl[:] = settle_ctrl
            else:
                self.data.ctrl[:] = lift_ctrl
            mujoco.mj_step(self.model, self.data)
            if t % frame_stride == 0:
                renderer.update_scene(self.data)
                frames.append(renderer.render().copy())

        output_gif.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output_gif, frames, fps=fps)
        return output_gif


def optimize_finger_controls(
    evaluator: Phase1GraspEvaluator,
    cfg: Phase1OptimizationConfig,
    initial_finger_ctrl: np.ndarray | None = None,
) -> dict[str, Any]:
    """Cross-entropy search for a high-quality grasp control vector."""
    rng = np.random.default_rng(cfg.seed)

    if initial_finger_ctrl is None:
        mean = evaluator.initial_ctrl[evaluator.finger_actuator_ids].astype(np.float64)
    else:
        mean = np.asarray(initial_finger_ctrl, dtype=np.float64)

    sigma = np.full_like(mean, cfg.sigma_init, dtype=np.float64)
    lo, hi = evaluator.finger_ctrl_min, evaluator.finger_ctrl_max

    elite_count = max(2, int(cfg.population * cfg.elite_fraction))

    best_score = -np.inf
    best_ctrl = mean.copy()
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []

    for it in range(cfg.iterations):
        samples = rng.normal(loc=mean, scale=sigma, size=(cfg.population, mean.size))
        samples = np.clip(samples, lo, hi)

        samples[0] = np.clip(mean, lo, hi)

        scores = np.zeros(cfg.population, dtype=np.float64)
        metrics_list: list[dict[str, float]] = []

        for i in range(cfg.population):
            s, m = evaluator.evaluate(samples[i])
            scores[i] = s
            metrics_list.append(m)

        elite_idx = np.argsort(scores)[-elite_count:]
        elite = samples[elite_idx]
        elite_scores = scores[elite_idx]

        mean = np.mean(elite, axis=0)
        sigma = np.std(elite, axis=0) + 1e-4

        iter_best_idx = int(np.argmax(scores))
        iter_best = float(scores[iter_best_idx])
        iter_mean = float(np.mean(scores))

        if iter_best > best_score:
            best_score = iter_best
            best_ctrl = samples[iter_best_idx].copy()
            best_metrics = metrics_list[iter_best_idx]

        history.append(
            {
                "iteration": float(it),
                "iter_best_score": iter_best,
                "iter_mean_score": iter_mean,
                "elite_mean_score": float(np.mean(elite_scores)),
                "best_score_so_far": float(best_score),
                "best_cube_lift_so_far": float(best_metrics.get("cube_lift", 0.0)),
                "best_contacts_so_far": float(best_metrics.get("cube_tip_contacts", 0.0)),
                "mean_sigma": float(np.mean(sigma)),
            }
        )

    return {
        "best_finger_ctrl": best_ctrl,
        "best_score": float(best_score),
        "best_metrics": best_metrics,
        "history": history,
    }
