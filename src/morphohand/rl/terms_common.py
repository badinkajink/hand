"""Shared helpers for the mjlab term modules (terms_obs / terms_reward / terms_event).

Package-private: the underscore names are imported by the sibling term modules and
re-exported through the `mjlab_terms` facade for back-compat.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

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


def _contact_force_mag(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    """Per-slot contact force magnitude for a contact sensor with a `force`
    field. Returns shape (num_envs, n_slots); the contact `force` field is a
    3-vector per slot, so we take its L2 norm. Zeros if unavailable."""
    sensor = env.scene.sensors[sensor_name]
    force = getattr(sensor.data, "force", None)
    if force is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    if force.dim() == 2:            # (B, 3) single slot
        return force.norm(dim=-1, keepdim=True)
    return force.norm(dim=-1)       # (B, n_slots, 3) -> (B, n_slots)


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


def _alignment_cos(env: "ManagerBasedRlEnv", object_name: str,
                    object_axis_local: tuple[float, float, float],
                    target_axis_world: tuple[float, float, float]) -> torch.Tensor:
    """Shared helper: cos(theta) between object's body-local axis (rotated
    into world frame) and the world target axis. Returns shape (num_envs,).
    Used by both `target_axis_alignment` (state reward), `target_axis_progress`
    (delta reward), and the velocity-based termination."""
    obj = env.scene[object_name]
    quat = obj.data.root_link_pose_w[:, 3:7]
    qw, qx, qy, qz = quat.unbind(dim=-1)
    ax, ay, az = object_axis_local
    bx = (1 - 2 * (qy * qy + qz * qz)) * ax + 2 * (qx * qy - qw * qz) * ay + 2 * (qx * qz + qw * qy) * az
    by = 2 * (qx * qy + qw * qz) * ax + (1 - 2 * (qx * qx + qz * qz)) * ay + 2 * (qy * qz - qw * qx) * az
    bz = 2 * (qx * qz - qw * qy) * ax + 2 * (qy * qz + qw * qx) * ay + (1 - 2 * (qx * qx + qy * qy)) * az
    tx, ty, tz = target_axis_world
    return (bx * tx + by * ty + bz * tz).clamp(-1.0, 1.0)


def _alignment_hold_counter(env: "ManagerBasedRlEnv", attr: str,
                            object_name: str,
                            object_axis_local: tuple[float, float, float],
                            target_axis_world: tuple[float, float, float],
                            align_thresh: float,
                            reorient_start_step: int) -> torch.Tensor:
    """Per-env count of *consecutive* policy steps with alignment cos >=
    `align_thresh` during the reorient phase. Resets on episode start and
    whenever alignment drops below threshold. Stored under `attr` so the
    success reward and success termination can keep independent (but
    identical) counters without double-incrementing a shared one.
    """
    cos = _alignment_cos(env, object_name, object_axis_local, target_axis_world)
    if not hasattr(env, attr):
        setattr(env, attr, torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
    counter = getattr(env, attr)
    just_started = env.episode_length_buf <= 1
    if just_started.any():
        counter[just_started] = 0
    in_phase = env.episode_length_buf >= int(reorient_start_step)
    aligned = (cos >= float(align_thresh)) & in_phase
    counter[aligned] += 1
    counter[~aligned] = 0
    return counter


def _contact_gate(env: "ManagerBasedRlEnv", sensor_name: str,
                   contact_gate_min: float) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found
    if found is None:
        return torch.zeros(env.num_envs, device=env.device)
    contact_mean = (found > 0).float().mean(dim=-1)
    return (contact_mean >= contact_gate_min).float()


def _in_lift_phase(env: "ManagerBasedRlEnv", lift_phase_start_step: int
                    ) -> torch.Tensor:
    return env.episode_length_buf >= int(lift_phase_start_step)
