"""LiftingCommand subclass that preserves a non-identity base quat at spawn.

mjlab's `LiftingCommand._resample_command` writes
`quat = quat_from_euler_xyz(0, 0, yaw)` — identity-yawed quat — directly to
the object's freejoint. That's fine for objects with identity rest quat (cube,
prism) but breaks flat-laying cylinders whose rest quat is a 90° X-rotation:
the override forces them to spawn standing vertical no matter what
`init_state.rot` we set on the EntityCfg.

This subclass takes a `base_quat` (wxyz) — typically the keyframe quat from
the foundational scene — and writes `quat_mul(yaw_world_z, base_quat)` so the
spawn pose is "rest quat applied first, then yaw rotation around world Z."
For identity `base_quat`, it's bit-identical to the parent behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.manipulation.mdp.commands import LiftingCommand, LiftingCommandCfg
from mjlab.utils.lab_api.math import (
    quat_from_euler_xyz, quat_mul, sample_uniform,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class LiftingCommandWithBaseQuat(LiftingCommand):
    cfg: "LiftingCommandWithBaseQuatCfg"

    def __init__(self, cfg: "LiftingCommandWithBaseQuatCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        bq = torch.tensor(cfg.base_quat, device=self.device, dtype=torch.float32)
        # broadcast to (num_envs, 4); _resample_command may be called with any subset
        self._base_quat_full = bq.unsqueeze(0).expand(self.num_envs, 4).contiguous()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)

        # --- target_pos (identical to parent) ---
        if self.cfg.difficulty == "fixed":
            target_pos = torch.tensor(
                [0.4, 0.0, 0.3], device=self.device, dtype=torch.float32
            ).expand(n, 3)
            self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]
        else:
            r = self.cfg.target_position_range
            lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
            upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
            target_pos = sample_uniform(lower, upper, (n, 3), device=self.device)
            self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]

        self.episode_success[env_ids] = 0.0

        # --- object pose (the override) ---
        if self.cfg.object_pose_range is not None:
            r = self.cfg.object_pose_range
            lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
            upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
            pos = sample_uniform(lower, upper, (n, 3), device=self.device)
            pos = pos + self._env.scene.env_origins[env_ids]

            yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
            yaw_quat = quat_from_euler_xyz(
                torch.zeros(n, device=self.device),
                torch.zeros(n, device=self.device),
                yaw,
            )
            # Compose: rotate first by base_quat (rest pose), then by world-Z yaw.
            # final = yaw_quat (world) * base_quat (intrinsic).
            base = self._base_quat_full[env_ids]
            quat = quat_mul(yaw_quat, base)
            # Optional spawn TILT jitter (roll/pitch) — handoff-robustness DR: lets the
            # object spawn slightly off-vertical/tilted so a reorient policy trained here
            # tolerates the varied lifted pose a real Policy-A lift delivers. Applied as a
            # small world-frame rotation on top of the yaw+base spawn quat.
            tlo, thi = self.cfg.spawn_tilt_range
            if thi > 0.0 or tlo < 0.0:
                roll = sample_uniform(tlo, thi, (n,), device=self.device)
                pitch = sample_uniform(tlo, thi, (n,), device=self.device)
                tilt_quat = quat_from_euler_xyz(roll, pitch, torch.zeros(n, device=self.device))
                quat = quat_mul(tilt_quat, quat)

            pose = torch.cat([pos, quat], dim=-1)
            velocity = torch.zeros(n, 6, device=self.device)
            self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
            self.object.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)


@dataclass(kw_only=True)
class LiftingCommandWithBaseQuatCfg(LiftingCommandCfg):
    base_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """wxyz quat applied to the object before the yaw sample. Identity =
    drop-in replacement for parent class."""
    spawn_tilt_range: tuple[float, float] = (0.0, 0.0)
    """Uniform roll/pitch jitter (rad) added to the spawn quat each reset.
    Handoff-robustness DR for skip-lift training. (0,0) = no tilt."""

    def build(self, env) -> LiftingCommandWithBaseQuat:
        return LiftingCommandWithBaseQuat(self, env)
