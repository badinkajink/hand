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

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


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


def object_lift_height(env: "ManagerBasedRlEnv",
                        object_name: str = "cube",
                        target_lift: float = 0.05) -> torch.Tensor:
    """Linear reward in clip(object_z - settle_z, 0, target_lift).

    `settle_z` is the cube's initial qpos z value at episode start
    (the default init state, written into model.key_qpos when the entity
    is built). We snapshot it lazily on first call and cache.
    """
    obj = env.scene[object_name]
    z = obj.data.root_link_pose_w[:, 2]
    if not hasattr(env, "_morphohand_settle_z"):
        env._morphohand_settle_z = z.detach().clone()
    settle_z = env._morphohand_settle_z
    lift = z - settle_z
    return lift.clamp(min=0.0, max=target_lift)


def object_drop_indicator(env: "ManagerBasedRlEnv",
                           object_name: str = "cube",
                           drop_threshold: float = 0.02) -> torch.Tensor:
    """1.0 when object_z fell below settle_z - drop_threshold, else 0.0."""
    obj = env.scene[object_name]
    z = obj.data.root_link_pose_w[:, 2]
    if not hasattr(env, "_morphohand_settle_z"):
        env._morphohand_settle_z = z.detach().clone()
    settle_z = env._morphohand_settle_z
    return (z < settle_z - drop_threshold).float()


def object_xy_drift(env: "ManagerBasedRlEnv",
                     object_name: str = "cube") -> torch.Tensor:
    """L2 drift of the object in the xy plane since episode start."""
    obj = env.scene[object_name]
    xy = obj.data.root_link_pose_w[:, :2]
    if not hasattr(env, "_morphohand_settle_xy"):
        env._morphohand_settle_xy = xy.detach().clone()
    return (xy - env._morphohand_settle_xy).norm(dim=-1)


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
