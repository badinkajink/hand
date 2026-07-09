"""Observation terms for the morphohand mjlab env (split from mjlab_terms.py)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from morphohand.rl.math import quat_mul, quat_rotate
from morphohand.rl.terms_common import _alignment_cos, _get_ref_batch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


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

    diff = obj_pos - palm_pos
    rel_pos = quat_rotate(palm_quat_inv, diff)
    rel_quat = quat_mul(palm_quat_inv, obj_quat)

    return torch.cat([rel_pos, rel_quat], dim=-1)


def target_axis_misalignment(env: "ManagerBasedRlEnv",
                              object_name: str = "cube",
                              object_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0),
                              target_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0)
                              ) -> torch.Tensor:
    """Raw angle (rad) between object's body-local axis and a world target
    axis. For observation use — gives the policy a vector to "drive to zero".
    Returns shape (num_envs,)."""
    cos_theta = _alignment_cos(env, object_name, object_axis_local, target_axis_world)
    return torch.acos(cos_theta).unsqueeze(-1)
