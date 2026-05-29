"""Custom mjlab RewTerm functions for morphohand cube grasping.

These adapt our pure-function reward terms (see `morphohand.rl.reward`) into
mjlab's RewTerm signature: `fn(env: ManagerBasedRlEnv, ...) -> torch.Tensor`
returning a `(num_envs,)` tensor.

Designed around the fact that mjlab's stock `staged_position_reward` is the
wrong signal for our morphology (it rewards palm getting close to the cube;
our hand grips with fingers from a static palm).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from morphohand.rl.reference_trajectory import ReferenceTrajectory

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_FINGER_JOINT_NAMES: tuple[str, ...] = (
    "thumb_yaw", "thumb_mcp", "thumb_pip",
    "index_yaw", "index_mcp", "index_pip",
    "middle_yaw", "middle_mcp", "middle_pip",
)


def _track(diff: torch.Tensor, alpha: float) -> torch.Tensor:
    return torch.exp(-alpha * diff.pow(2).mean(dim=-1))


def _get_step_dt(env: "ManagerBasedRlEnv") -> float:
    if hasattr(env, "step_dt"):
        return float(env.step_dt)
    sim = getattr(env, "sim", None)
    sim_dt = float(getattr(sim, "dt", 0.002)) if sim is not None else 0.002
    decimation = float(getattr(env, "decimation", 1))
    return sim_dt * decimation


def _get_env_time(env: "ManagerBasedRlEnv") -> torch.Tensor:
    if hasattr(env, "progress_buf"):
        return env.progress_buf.to(dtype=torch.float32) * _get_step_dt(env)
    return torch.zeros(env.num_envs, device=env.device)


def _get_ref(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str
             ) -> ReferenceTrajectory:
    if not hasattr(env, "_morphohand_ref"):
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(frozen_scene_xml))
        env._morphohand_ref = ReferenceTrajectory.from_run_dir(run_dir, model)
    return env._morphohand_ref


def _get_ref_batch(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str
                   ) -> dict[str, torch.Tensor]:
    ref = _get_ref(env, run_dir, frozen_scene_xml)
    ts = _get_env_time(env).detach().cpu().numpy()
    batch = ref.batch_at(ts)
    return {k: torch.as_tensor(v, device=env.device, dtype=torch.float32) for k, v in batch.items()}


def _get_finger_joint_ids(env: "ManagerBasedRlEnv") -> list[int]:
    if not hasattr(env, "_morphohand_finger_joint_ids"):
        robot = env.scene["robot"]
        names = list(robot.joint_names)
        env._morphohand_finger_joint_ids = [names.index(n) for n in _FINGER_JOINT_NAMES]
    return env._morphohand_finger_joint_ids


def _get_finger_qpos(env: "ManagerBasedRlEnv") -> torch.Tensor:
    robot = env.scene["robot"]
    ids = _get_finger_joint_ids(env)
    return robot.data.joint_pos[:, ids]


def _get_finger_action(env: "ManagerBasedRlEnv") -> torch.Tensor:
    action = None
    if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
        action = env.action_manager.action
    elif hasattr(env, "_last_actions"):
        action = env._last_actions
    if action is None:
        return torch.zeros((env.num_envs, len(_FINGER_JOINT_NAMES)), device=env.device)
    if action.dim() == 1:
        action = action.unsqueeze(0)
    if action.shape[-1] > len(_FINGER_JOINT_NAMES):
        action = action[:, :len(_FINGER_JOINT_NAMES)]
    return action


def track_finger_qpos(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str,
                       alpha: float = 20.0) -> torch.Tensor:
    ref = _get_ref_batch(env, run_dir, frozen_scene_xml)
    finger_qpos = _get_finger_qpos(env)
    return _track(finger_qpos - ref["finger_qpos"], alpha)


def track_object_pos(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str,
                      alpha: float = 200.0) -> torch.Tensor:
    ref = _get_ref_batch(env, run_dir, frozen_scene_xml)
    obj = env.scene["cube"]
    pos = obj.data.root_link_pose_w[:, :3]
    return _track(pos - ref["object_pos"], alpha)


def track_object_quat(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str,
                       alpha: float = 10.0) -> torch.Tensor:
    ref = _get_ref_batch(env, run_dir, frozen_scene_xml)
    obj = env.scene["cube"]
    quat = obj.data.root_link_pose_w[:, 3:7]
    dot = torch.sum(quat * ref["object_quat"], dim=-1)
    geo_sq = torch.clamp(1.0 - dot * dot, min=0.0)
    return torch.exp(-alpha * geo_sq)


def track_finger_ctrl_anchor(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str,
                              alpha: float = 4.0) -> torch.Tensor:
    """Reward staying close to the CEM finger ctrl.

    The finger action is configured with default_offset = CEM best_finger_ctrl,
    so the raw policy action *is* the residual from the reference ctrl.
    Penalising its magnitude is equivalent to anchoring the absolute ctrl
    to the reference. (`ref["finger_ctrl"]` is not subtracted again — that
    would double-count the offset and saturate the reward to ~0.)
    """
    del run_dir, frozen_scene_xml  # ref isn't needed here; kept for cfg parity
    action = _get_finger_action(env)
    return _track(action, alpha)


def ref_finger_qpos(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str
                    ) -> torch.Tensor:
    ref = _get_ref_batch(env, run_dir, frozen_scene_xml)
    return ref["finger_qpos"]


def ref_object_pose(env: "ManagerBasedRlEnv", run_dir: str, frozen_scene_xml: str
                    ) -> torch.Tensor:
    ref = _get_ref_batch(env, run_dir, frozen_scene_xml)
    return torch.cat([ref["object_pos"], ref["object_quat"]], dim=-1)


def object_pose_rel_palm(env: "ManagerBasedRlEnv",
                          object_name: str = "cube",
                          palm_body: str = "palm_pose") -> torch.Tensor:
    """Actual cube pose (pos+quat, 7-d) expressed in the palm frame.

    Adds a true observation of where the cube actually is — distinct from
    `ref_object_pose` which is a fixed CEM trajectory. Critical under
    domain randomization (cube spawn xy/yaw jitter): without this, the
    policy can't observe how the cube has shifted from its nominal pose.

    Returns shape (num_envs, 7): [px, py, pz, qw, qx, qy, qz].
    """
    robot = env.scene["robot"]
    obj = env.scene[object_name]
    palm_id = robot.body_names.index(palm_body)

    palm_pose_w = robot.data.body_link_pose_w[:, palm_id, :]  # (B, 7)
    obj_pose_w = obj.data.root_link_pose_w  # (B, 7)

    # pos in palm frame: rotate (obj_pos - palm_pos) by palm_quat^-1
    palm_pos = palm_pose_w[:, :3]
    palm_quat = palm_pose_w[:, 3:7]  # wxyz
    obj_pos = obj_pose_w[:, :3]
    obj_quat = obj_pose_w[:, 3:7]

    # Inverse quat: conjugate (negate xyz). wxyz format.
    palm_quat_inv = palm_quat.clone()
    palm_quat_inv[:, 1:] *= -1.0

    # Rotate (obj_pos - palm_pos) by palm_quat_inv.
    # quat rotation: q * v * q^-1, treating v as pure quat (0, vx, vy, vz).
    diff = obj_pos - palm_pos
    rel_pos = _quat_rotate(palm_quat_inv, diff)
    rel_quat = _quat_mul(palm_quat_inv, obj_quat)

    return torch.cat([rel_pos, rel_quat], dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two batched wxyz quaternions, shapes (B, 4)."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors `v` (B, 3) by batched wxyz quats `q` (B, 4)."""
    qw, qx, qy, qz = q.unbind(dim=-1)
    vx, vy, vz = v.unbind(dim=-1)
    # q * (0, v) * q^-1
    # Use the explicit formula: v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    qvec = torch.stack([qx, qy, qz], dim=-1)
    t = 2.0 * torch.cross(qvec, v, dim=-1)
    return v + qw.unsqueeze(-1) * t + torch.cross(qvec, t, dim=-1)


def fingertip_contact_mean(env: "ManagerBasedRlEnv",
                            sensor_name: str = "fingertip_cube_contact") -> torch.Tensor:
    """Mean fingertip-cube contact across the 3 tips, per env.

    Returns shape (num_envs,). Range [0, 1]: 1.0 means all three tips in
    contact with the cube; 0.0 means no contact.
    """
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (B, N=3)
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (found > 0).float().mean(dim=-1)


def fingertip_contact_min(env: "ManagerBasedRlEnv",
                           sensor_name: str = "fingertip_cube_contact") -> torch.Tensor:
    """Worst-finger contact, per env. Discourages 2-finger grips.

    Returns shape (num_envs,) in [0, 1].
    """
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (found > 0).float().min(dim=-1).values


def _spawn_pose(env: "ManagerBasedRlEnv", object_name: str = "cube"):
    """Per-env cube spawn pose, refreshed on episode reset.

    Returns dict of tensors {xy: (B, 2), z: (B,), quat: (B, 4)} taken at
    the FIRST step of each episode (episode_length_buf == 1).

    The previous implementation cached on first call across all envs and
    never refreshed — so under DR, where the spawn varies per episode,
    the "drift from spawn" was actually drift from the very first
    episode's pose, which is meaningless after the first reset.
    """
    obj = env.scene[object_name]
    pose = obj.data.root_link_pose_w  # (B, 7)
    if not hasattr(env, "_morphohand_spawn_pose"):
        env._morphohand_spawn_pose = pose.detach().clone()  # init buffer

    # Snapshot pose for envs that just started a new episode.
    # episode_length_buf is incremented after the env step; == 1 means we
    # just finished the first step after reset. Capture HERE (after one
    # step of settling so the cube isn't penetrating the floor).
    just_started = env.episode_length_buf <= 1
    if just_started.any():
        env._morphohand_spawn_pose[just_started] = pose[just_started].detach().clone()
    return env._morphohand_spawn_pose


def object_lift_height(env: "ManagerBasedRlEnv",
                        object_name: str = "cube",
                        target_lift: float = 0.05) -> torch.Tensor:
    """Linear reward in clip(object_z - settle_z, 0, target_lift). Per-env
    settle_z refreshed every episode."""
    obj = env.scene[object_name]
    z = obj.data.root_link_pose_w[:, 2]
    settle_z = _spawn_pose(env, object_name)[:, 2]
    return (z - settle_z).clamp(min=0.0, max=target_lift)


def object_drop_indicator(env: "ManagerBasedRlEnv",
                           object_name: str = "cube",
                           drop_threshold: float = 0.02) -> torch.Tensor:
    """1.0 when object_z fell below spawn_z - drop_threshold, else 0.0.
    Per-env spawn_z refreshed every episode."""
    obj = env.scene[object_name]
    z = obj.data.root_link_pose_w[:, 2]
    settle_z = _spawn_pose(env, object_name)[:, 2]
    return (z < settle_z - drop_threshold).float()


def object_xy_drift(env: "ManagerBasedRlEnv",
                     object_name: str = "cube") -> torch.Tensor:
    """L2 drift of the object in the xy plane since episode start."""
    obj = env.scene[object_name]
    xy = obj.data.root_link_pose_w[:, :2]
    spawn_xy = _spawn_pose(env, object_name)[:, :2]
    return (xy - spawn_xy).norm(dim=-1)


def object_orientation_drift(env: "ManagerBasedRlEnv",
                              object_name: str = "cube") -> torch.Tensor:
    """Geodesic quat distance (rad) between current and spawn orientation.

    Penalizes the object rotating from its spawn orientation — what we want
    for a "lift only" trajectory. 0 = no rotation, pi = flipped 180 deg.
    """
    obj = env.scene[object_name]
    quat = obj.data.root_link_pose_w[:, 3:7]
    spawn_quat = _spawn_pose(env, object_name)[:, 3:7]
    dot = (quat * spawn_quat).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def finger_drift_from_grip(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """L2 distance between current finger qpos and the finger grip ctrl
    (CEM `best_finger_ctrl`, which is the default_offset on the finger
    action term). Penalizes the policy deviating from the trained grip
    pose, encouraging stable contact configuration."""
    finger_jids = _get_finger_joint_ids(env)
    robot = env.scene["robot"]
    qpos = robot.data.joint_pos[:, finger_jids]  # (B, 9)
    # Pull target from the action term's offset (per-env, set at build time)
    a_term = env.action_manager.get_term("finger_ctrl")
    target = a_term._target if hasattr(a_term, "_target") else None
    if target is None:
        return torch.zeros(env.num_envs, device=env.device)
    return (qpos - target.unsqueeze(0)).norm(dim=-1)


def anneal_tracking_weights(env: "ManagerBasedRlEnv", env_ids,
                              term_names: tuple[str, ...],
                              base_weights: tuple[float, ...],
                              final_scale: float,
                              anneal_iters: int,
                              num_steps_per_env: int = 24) -> float:
    """Linearly scale the weights of named reward terms from `base_weights`
    at iter 0 down to `final_scale * base_weights` at `anneal_iters`.

    Use to phase out tracking-from-CEM rewards over training: early on
    tracking keeps the policy in the basin; later it's a misleading signal
    under DR (the CEM reference object_pos doesn't match the spawned cube
    position).
    """
    del env_ids
    if anneal_iters <= 0 or not term_names:
        return 1.0
    iters = int(env.common_step_counter) // max(1, int(num_steps_per_env))
    progress = float(min(1.0, max(0.0, iters / float(anneal_iters))))
    scale = 1.0 + (float(final_scale) - 1.0) * progress
    for name, base in zip(term_names, base_weights, strict=False):
        try:
            cfg = env.reward_manager.get_term_cfg(name)
        except (ValueError, AttributeError):
            continue
        cfg.weight = float(base) * scale
    return scale


def anneal_cube_spawn_jitter(env: "ManagerBasedRlEnv", env_ids,
                              x_max: float, y_max: float, yaw_max: float,
                              anneal_iters: int,
                              x_center: float = 0.0, y_center: float = 0.0,
                              num_steps_per_env: int = 24,
                              command_name: str = "lift_height") -> float:
    """Linearly ramp the cube spawn jitter on `command_name`'s
    `object_pose_range` from 0 to ±(x_max, y_max, yaw_max) around
    (x_center, y_center) over the first `anneal_iters` PPO iterations.
    """
    del env_ids
    if anneal_iters <= 0:
        return 1.0
    iters = int(env.common_step_counter) // max(1, int(num_steps_per_env))
    progress = float(min(1.0, max(0.0, iters / float(anneal_iters))))

    cmd = env.command_manager.get_term(command_name)
    cmd.cfg.object_pose_range.x = (x_center - x_max * progress, x_center + x_max * progress)
    cmd.cfg.object_pose_range.y = (y_center - y_max * progress, y_center + y_max * progress)
    cmd.cfg.object_pose_range.yaw = (-yaw_max * progress, yaw_max * progress)
    return progress


# ----------------------------------------------------------------------
# Contact-gated stability rewards
# ----------------------------------------------------------------------
# These multiply the raw drift signal by a contact gate so the penalty
# only fires once at least `contact_gate_min` fraction of tips are touching
# the object. Rationale: during approach the fingers are clear of the cube
# and the cube shouldn't have moved at all (drift ~0 anyway). After
# first-contact the cube physically WILL be perturbed by the closing
# motion — penalising that legitimate perturbation pushes the policy to
# never touch the cube. Gating concentrates the credit on "once you HAVE
# the cube, hold it still".


def _contact_gate(env: "ManagerBasedRlEnv", sensor_name: str,
                   contact_gate_min: float) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    contact_mean = (found > 0).float().mean(dim=-1)
    return (contact_mean >= contact_gate_min).float()


def object_xy_drift_gated(env: "ManagerBasedRlEnv",
                            object_name: str = "cube",
                            contact_gate_min: float = 0.5,
                            sensor_name: str = "fingertip_cube_contact"
                            ) -> torch.Tensor:
    return object_xy_drift(env, object_name) * _contact_gate(env, sensor_name, contact_gate_min)


def object_orientation_drift_gated(env: "ManagerBasedRlEnv",
                                     object_name: str = "cube",
                                     contact_gate_min: float = 0.5,
                                     sensor_name: str = "fingertip_cube_contact"
                                     ) -> torch.Tensor:
    return object_orientation_drift(env, object_name) * _contact_gate(env, sensor_name, contact_gate_min)


def finger_drift_from_grip_gated(env: "ManagerBasedRlEnv",
                                   contact_gate_min: float = 0.5,
                                   sensor_name: str = "fingertip_cube_contact"
                                   ) -> torch.Tensor:
    return finger_drift_from_grip(env) * _contact_gate(env, sensor_name, contact_gate_min)


# ----------------------------------------------------------------------
# Lift-phase early terminations
# ----------------------------------------------------------------------
# Engaged once `env.episode_length_buf >= lift_phase_start_step` (i.e.,
# after the scripted lift ramp has completed + a few steps of hold). These
# do NOT fire during approach or initial contact formation, so the policy
# isn't punished for legitimate transients. Returning True for an env
# causes mjlab to reset that env at end-of-step; GAE treats it as a
# terminal state (no bootstrap), which is the negative signal.


def _in_lift_phase(env: "ManagerBasedRlEnv", lift_phase_start_step: int
                    ) -> torch.Tensor:
    return env.episode_length_buf >= int(lift_phase_start_step)


def terminate_object_slip(env: "ManagerBasedRlEnv",
                            lift_phase_start_step: int = 40,
                            xy_drift_threshold: float = 0.015,
                            object_name: str = "cube") -> torch.Tensor:
    drift = object_xy_drift(env, object_name)
    return _in_lift_phase(env, lift_phase_start_step) & (drift > xy_drift_threshold)


def terminate_object_orientation_slip(env: "ManagerBasedRlEnv",
                                        lift_phase_start_step: int = 40,
                                        orientation_drift_threshold: float = 0.5,
                                        object_name: str = "cube"
                                        ) -> torch.Tensor:
    drift = object_orientation_drift(env, object_name)
    return _in_lift_phase(env, lift_phase_start_step) & (drift > orientation_drift_threshold)


def terminate_object_drop(env: "ManagerBasedRlEnv",
                            lift_phase_start_step: int = 40,
                            drop_threshold: float = 0.02,
                            object_name: str = "cube") -> torch.Tensor:
    obj = env.scene[object_name]
    z = obj.data.root_link_pose_w[:, 2]
    spawn_z = _spawn_pose(env, object_name)[:, 2]
    return _in_lift_phase(env, lift_phase_start_step) & (z < spawn_z - drop_threshold)


def terminate_tip_lost(env: "ManagerBasedRlEnv",
                         lift_phase_start_step: int = 40,
                         consecutive_steps: int = 3,
                         sensor_name: str = "fingertip_cube_contact"
                         ) -> torch.Tensor:
    """Terminate if any tip is off the object for >= consecutive_steps
    consecutive policy steps during the lift phase.

    Per-env counter persists across steps; resets when (a) the env
    resets, or (b) contact is restored on all tips.
    """
    if not hasattr(env, "_morphohand_tip_lost_counter"):
        env._morphohand_tip_lost_counter = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )
    counter = env._morphohand_tip_lost_counter

    just_started = env.episode_length_buf <= 1
    if just_started.any():
        counter[just_started] = 0

    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    any_tip_lost = (found <= 0).any(dim=-1)
    in_phase = _in_lift_phase(env, lift_phase_start_step)

    fire = any_tip_lost & in_phase
    counter[fire] += 1
    counter[~fire] = 0
    return counter >= int(consecutive_steps)


def terminate_finger_slip(env: "ManagerBasedRlEnv",
                            lift_phase_start_step: int = 40,
                            finger_drift_threshold: float = 0.3
                            ) -> torch.Tensor:
    drift = finger_drift_from_grip(env)
    return _in_lift_phase(env, lift_phase_start_step) & (drift > finger_drift_threshold)


def fingertip_to_object_distance(env: "ManagerBasedRlEnv",
                                   object_name: str = "cube",
                                   fingertip_body_names: tuple[str, ...] = (
                                       "thumb_tip", "index_tip", "middle_tip",
                                   )) -> torch.Tensor:
    """Sum of fingertip-to-object distances (m), per env.

    Used as a penalty term (smaller is better). When all three tips are
    on the cube surface, returns ~3 * cube_size + slack.
    """
    robot = env.scene["robot"]
    obj = env.scene[object_name]
    obj_pos = obj.data.root_link_pose_w[:, :3]  # (B, 3)
    body_ids = []
    for name in fingertip_body_names:
        bid = robot.body_names.index(name) if name in robot.body_names else -1
        if bid >= 0:
            body_ids.append(bid)
    if not body_ids:
        return torch.zeros(env.num_envs, device=env.device)
    tip_pos = robot.data.body_link_pose_w[:, body_ids, :3]  # (B, 3, 3)
    diff = tip_pos - obj_pos.unsqueeze(1)  # (B, 3, 3)
    dist = diff.norm(dim=-1)  # (B, 3)
    return dist.sum(dim=-1)
