from __future__ import annotations
# pyright: reportMissingImports=false

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
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


@dataclass
class Phase1AutodiffConfig:
    iterations: int = 80
    learning_rate: float = 0.04
    grad_clip_norm: float = 5.0
    contact_distance_threshold: float = 0.01
    contact_distance_sharpness: float = 0.003
    score_weight_contact_proxy: float = 0.4
    score_weight_ctrl_l2: float = 0.02
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
        self.cube_half_extents = self._infer_cube_half_extents()

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

    def _infer_cube_half_extents(self) -> np.ndarray:
        # Read size directly from the cube geom so non-cube prisms are handled correctly.
        box_geoms = []
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) != self.cube_body_id:
                continue
            if int(self.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            box_geoms.append(geom_id)

        if not box_geoms:
            raise ValueError("Cube body does not contain a box geom")

        geom_id = box_geoms[0]
        return np.asarray(self.model.geom_size[geom_id, :3], dtype=np.float64)

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


def optimize_finger_controls_autodiff(
    evaluator: Phase1GraspEvaluator,
    cfg: Phase1AutodiffConfig,
    initial_finger_ctrl: np.ndarray | None = None,
) -> dict[str, Any]:
    """MJX-native autodiff optimizer for Phase 1 finger controls.

    The differentiable score is a smooth proxy aligned to the simulation objective,
    while final reporting still uses the standard MuJoCo evaluator metrics.
    """
    rng = np.random.default_rng(cfg.seed)

    if initial_finger_ctrl is None:
        q = evaluator.initial_ctrl[evaluator.finger_actuator_ids].astype(np.float64)
    else:
        q = np.asarray(initial_finger_ctrl, dtype=np.float64)

    # Small random perturbation avoids perfectly symmetric starts with zero gradient.
    q = q + rng.normal(0.0, 1e-4, size=q.shape)

    lo = evaluator.finger_ctrl_min.astype(np.float64)
    hi = evaluator.finger_ctrl_max.astype(np.float64)

    evaluator._reset_to_keyframe()
    base_ctrl = evaluator.initial_ctrl.astype(np.float32)
    mjx_model = mjx.put_model(evaluator.model)
    mjx_data0 = mjx.put_data(evaluator.model, evaluator.data)

    finger_ids = jnp.asarray(evaluator.finger_actuator_ids, dtype=jnp.int32)
    palm_pz_id = int(evaluator.pose_actuator_ids[2])
    base_ctrl_j = jnp.asarray(base_ctrl, dtype=jnp.float32)
    lo_j = jnp.asarray(lo, dtype=jnp.float32)
    hi_j = jnp.asarray(hi, dtype=jnp.float32)

    tip_ids = jnp.asarray(evaluator.tip_body_ids, dtype=jnp.int32)
    cube_body_id = int(evaluator.cube_body_id)
    cube_half_extents = jnp.asarray(evaluator.cube_half_extents, dtype=jnp.float32)

    settle_steps = int(evaluator.cfg.settle_steps)
    lift_steps = int(evaluator.cfg.lift_steps)
    hold_steps = int(evaluator.cfg.hold_steps)
    weight_dist = float(evaluator.cfg.objective_weight_distance)
    weight_lift = float(evaluator.cfg.objective_weight_lift)
    weight_vel = float(evaluator.cfg.objective_weight_velocity_penalty)
    lift_target = float(evaluator.initial_palm_pz_target + evaluator.cfg.lift_delta_z)

    threshold = float(cfg.contact_distance_threshold)
    sharpness = float(cfg.contact_distance_sharpness)
    weight_contact = float(cfg.score_weight_contact_proxy)
    weight_ctrl_l2 = float(cfg.score_weight_ctrl_l2)

    def _step_n(data: Any, ctrl: jax.Array, n_steps: int) -> Any:
        def body_fn(_, state):
            state = state.replace(ctrl=ctrl)
            return mjx.step(mjx_model, state)

        return jax.lax.fori_loop(0, n_steps, body_fn, data)

    @jax.jit
    def _surrogate_score(finger_ctrl: jax.Array) -> jax.Array:
        finger_ctrl = jnp.clip(finger_ctrl, lo_j, hi_j)

        settle_ctrl = base_ctrl_j.at[finger_ids].set(finger_ctrl)
        lift_ctrl = settle_ctrl.at[palm_pz_id].set(lift_target)

        data = _step_n(mjx_data0, settle_ctrl, settle_steps)
        cube_pos = data.xpos[cube_body_id]
        cube_rot = data.xmat[cube_body_id].reshape(3, 3)
        tip_world = data.xpos[tip_ids]

        local = (tip_world - cube_pos) @ cube_rot
        clipped = jnp.clip(local, -cube_half_extents, cube_half_extents)
        closest_world = clipped @ cube_rot.T + cube_pos
        distances = jnp.linalg.norm(tip_world - closest_world, axis=1)

        contact_proxy = jnp.sum(jax.nn.sigmoid((threshold - distances) / sharpness))
        mean_dist = jnp.mean(distances)
        z_before = data.xpos[cube_body_id, 2]

        def lift_scan(carry, _):
            state, z_peak = carry
            state = state.replace(ctrl=lift_ctrl)
            state = mjx.step(mjx_model, state)
            z_peak = jnp.maximum(z_peak, state.xpos[cube_body_id, 2])
            return (state, z_peak), z_peak

        (data, z_peak), _ = jax.lax.scan(lift_scan, (data, z_before), xs=None, length=lift_steps)

        data = _step_n(data, lift_ctrl, hold_steps)
        cube_vel_norm = jnp.linalg.norm(data.qvel[:6])

        lift_amount = z_peak - z_before
        ctrl_l2 = jnp.mean(jnp.square(finger_ctrl))

        return (
            weight_lift * lift_amount
            - weight_dist * mean_dist
            + weight_contact * contact_proxy
            - weight_vel * cube_vel_norm
            - weight_ctrl_l2 * ctrl_l2
        )

    score_fn = jax.jit(_surrogate_score)
    grad_fn = jax.jit(jax.jacfwd(_surrogate_score))

    best_score = -np.inf
    best_ctrl = q.copy()
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []

    q_j = jnp.asarray(q, dtype=jnp.float32)
    for it in range(cfg.iterations):
        surrogate_value = score_fn(q_j)
        grad = grad_fn(q_j)
        grad_norm = jnp.linalg.norm(grad)
        scale = jnp.maximum(1.0, grad_norm / cfg.grad_clip_norm)
        grad = grad / scale

        q_j = q_j + cfg.learning_rate * grad
        q_j = jnp.clip(q_j, lo_j, hi_j)

        q_np = np.asarray(q_j, dtype=np.float64)
        sim_score, sim_metrics = evaluator.evaluate(q_np)

        if sim_score > best_score:
            best_score = float(sim_score)
            best_ctrl = q_np.copy()
            best_metrics = sim_metrics

        history.append(
            {
                "iteration": float(it),
                "iter_best_score": float(sim_score),
                "iter_mean_score": float(sim_score),
                "elite_mean_score": float(surrogate_value),
                "best_score_so_far": float(best_score),
                "best_cube_lift_so_far": float(best_metrics.get("cube_lift", 0.0)),
                "best_contacts_so_far": float(best_metrics.get("cube_tip_contacts", 0.0)),
                "mean_sigma": float(grad_norm),
            }
        )

    return {
        "best_finger_ctrl": best_ctrl,
        "best_score": float(best_score),
        "best_metrics": best_metrics,
        "history": history,
    }
