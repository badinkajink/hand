from __future__ import annotations
# pyright: reportMissingImports=false

from dataclasses import dataclass
import time
from typing import Any

import jax
import jax.numpy as jnp
import mujoco.mjx as mjx
import numpy as np

from .phase1_common import Phase1GraspEvaluator


@dataclass
class Phase1DiffMJXMVPConfig:
    iterations: int = 80
    learning_rate: float = 0.03
    grad_clip_norm: float = 5.0
    contact_distance_threshold: float = 0.01
    contact_distance_sharpness: float = 0.003
    score_weight_contact_proxy: float = 0.8
    score_weight_ctrl_l2: float = 0.015
    eval_interval: int = 1
    seed: int = 0


def optimize_finger_controls_diffmjx_mvp(
    evaluator: Phase1GraspEvaluator,
    cfg: Phase1DiffMJXMVPConfig,
    initial_finger_ctrl: np.ndarray | None = None,
) -> dict[str, Any]:
    """DiffMJX MVP lane using contacts-from-distance style smooth rewards."""
    rng = np.random.default_rng(cfg.seed)

    if initial_finger_ctrl is None:
        q = evaluator.initial_ctrl[evaluator.finger_actuator_ids].astype(np.float64)
    else:
        q = np.asarray(initial_finger_ctrl, dtype=np.float64)

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
    total_dynamic_steps = max(1, lift_steps + hold_steps)

    weight_dist = float(evaluator.cfg.objective_weight_distance)
    weight_lift = float(evaluator.cfg.objective_weight_lift)
    weight_vel = float(evaluator.cfg.objective_weight_velocity_penalty)
    weight_xy = float(evaluator.cfg.objective_weight_xy_drift_penalty)
    weight_drop = float(evaluator.cfg.objective_weight_drop_penalty)

    threshold = float(cfg.contact_distance_threshold)
    sharpness = float(cfg.contact_distance_sharpness)
    weight_contact = float(cfg.score_weight_contact_proxy)
    weight_ctrl_l2 = float(cfg.score_weight_ctrl_l2)

    def _soft_contact_proxy(distances: jax.Array) -> jax.Array:
        logits = jnp.clip((threshold - distances) / sharpness, -40.0, 40.0)
        return jax.nn.sigmoid(logits)

    def _step_n(data: Any, ctrl: jax.Array, n_steps: int) -> Any:
        def body_fn(state, _):
            state = state.replace(ctrl=ctrl)
            return mjx.step(mjx_model, state), 0.0

        data, _ = jax.lax.scan(body_fn, data, xs=None, length=n_steps)
        return data

    def _soft_tip_distances(data: Any) -> jax.Array:
        cube_pos = data.xpos[cube_body_id]
        cube_rot = data.xmat[cube_body_id].reshape(3, 3)
        tip_world = data.xpos[tip_ids]
        local = (tip_world - cube_pos) @ cube_rot
        clipped = jnp.clip(local, -cube_half_extents, cube_half_extents)
        closest_world = clipped @ cube_rot.T + cube_pos
        return jnp.linalg.norm(tip_world - closest_world, axis=1)

    @jax.jit
    def _surrogate_score(finger_ctrl: jax.Array) -> jax.Array:
        finger_ctrl = jnp.clip(finger_ctrl, lo_j, hi_j)

        settle_ctrl = base_ctrl_j.at[finger_ids].set(finger_ctrl)
        data = _step_n(mjx_data0, settle_ctrl, settle_steps)

        distances = _soft_tip_distances(data)
        mean_dist = jnp.mean(distances)
        contact_proxy_settle = jnp.sum(_soft_contact_proxy(distances))

        z_before = data.xpos[cube_body_id, 2]
        xy_before = data.xpos[cube_body_id, :2]

        def dyn_scan(carry, t):
            state, z_peak, persist_acc = carry
            ramp_denom = jnp.maximum(1, jnp.asarray(evaluator.cfg.lift_ramp_steps, dtype=jnp.int32))
            lift_scale = jnp.where(
                t < lift_steps,
                jnp.minimum(1.0, (t + 1).astype(jnp.float32) / ramp_denom.astype(jnp.float32)),
                1.0,
            )
            ctrl_t = settle_ctrl.at[palm_pz_id].set(
                evaluator.initial_palm_pz_target + lift_scale * evaluator.cfg.lift_delta_z
            )
            state = state.replace(ctrl=ctrl_t)
            state = mjx.step(mjx_model, state)

            d = _soft_tip_distances(state)
            soft_contacts = _soft_contact_proxy(d)
            persist_acc = persist_acc + jnp.min(soft_contacts)
            z_peak = jnp.maximum(z_peak, state.xpos[cube_body_id, 2])
            return (state, z_peak, persist_acc), 0.0

        (data, z_peak, persist_acc), _ = jax.lax.scan(
            dyn_scan,
            (data, z_before, jnp.asarray(0.0, dtype=jnp.float32)),
            xs=jnp.arange(total_dynamic_steps, dtype=jnp.int32),
            length=total_dynamic_steps,
        )

        z_after = data.xpos[cube_body_id, 2]
        xy_after = data.xpos[cube_body_id, :2]
        cube_vel_norm = jnp.linalg.norm(data.qvel[:6])

        lift_amount = z_peak - z_before
        xy_drift = jnp.linalg.norm(xy_after - xy_before)
        z_drop = jnp.maximum(0.0, z_peak - z_after)
        persistence = persist_acc / float(total_dynamic_steps)
        ctrl_l2 = jnp.mean(jnp.square(finger_ctrl))

        return (
            weight_lift * lift_amount
            - weight_dist * mean_dist
            + weight_contact * (contact_proxy_settle + persistence)
            - weight_vel * cube_vel_norm
            - weight_xy * xy_drift
            - weight_drop * z_drop
            - weight_ctrl_l2 * ctrl_l2
        )

    score_only_fn = jax.jit(_surrogate_score)
    rev_score_and_grad_fn = jax.jit(jax.value_and_grad(_surrogate_score))
    fwd_grad_fn = jax.jit(jax.jacfwd(_surrogate_score))

    best_score = -np.inf
    best_ctrl = q.copy()
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []
    optimization_start = time.perf_counter()

    q_j = jnp.asarray(q, dtype=jnp.float32)
    eval_interval = max(1, int(cfg.eval_interval))
    grad_mode = "reverse"

    for it in range(cfg.iterations):
        iteration_start = time.perf_counter()
        try:
            surrogate_value, grad = rev_score_and_grad_fn(q_j)
        except Exception as exc:
            if grad_mode != "forward":
                # Some MJX kernels do not expose reverse-mode rules on all GPU builds.
                print(f"DiffMJX reverse-mode gradient unavailable, falling back to forward-mode jacfwd: {exc}")
                grad_mode = "forward"
            surrogate_value = score_only_fn(q_j)
            grad = fwd_grad_fn(q_j)

        grad_norm = jnp.linalg.norm(grad)
        scale = jnp.maximum(1.0, grad_norm / cfg.grad_clip_norm)
        grad = grad / scale

        q_j = q_j + cfg.learning_rate * grad
        q_j = jnp.clip(q_j, lo_j, hi_j)

        q_np = np.asarray(q_j, dtype=np.float64)
        sim_score = float("nan")
        sim_metrics: dict[str, float] = {}
        if it % eval_interval == 0 or it == cfg.iterations - 1:
            sim_score, sim_metrics = evaluator.evaluate(q_np)
            if sim_score > best_score:
                best_score = float(sim_score)
                best_ctrl = q_np.copy()
                best_metrics = sim_metrics

        history.append(
            {
                "iteration": float(it),
                "iter_best_score": float(sim_score) if not np.isnan(sim_score) else float(best_score),
                "iter_mean_score": float(sim_score) if not np.isnan(sim_score) else float(best_score),
                "elite_mean_score": float(surrogate_value),
                "best_score_so_far": float(best_score),
                "best_cube_lift_so_far": float(best_metrics.get("cube_lift", 0.0)),
                "best_contacts_so_far": float(best_metrics.get("cube_tip_contacts", 0.0)),
                "mean_sigma": float(grad_norm),
                "iteration_seconds": float(time.perf_counter() - iteration_start),
            }
        )

    optimization_seconds = float(time.perf_counter() - optimization_start)
    mean_iteration_seconds = (
        float(np.mean([row["iteration_seconds"] for row in history])) if history else 0.0
    )

    return {
        "best_finger_ctrl": best_ctrl,
        "best_score": float(best_score),
        "best_metrics": best_metrics,
        "history": history,
        "optimization_wall_time_seconds": optimization_seconds,
        "mean_iteration_seconds": mean_iteration_seconds,
    }
