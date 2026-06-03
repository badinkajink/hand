"""Custom mjlab ActionTerms for morphohand.

`ScriptedPalmAction` replays the CEM palm schedule (constant px/py/rx/ry/rz at
the keyframe values, with palm_pz ramping linearly during the lift phase).
The action term consumes zero policy outputs (`action_dim = 0`) and writes
joint position targets directly on every sim substep.

Why scripted: the CEM rollout's only palm motion is a single `palm_pz` ramp.
Letting PPO discover that schedule end-to-end is wasteful — the plan
(witty-tumbling-flamingo.md, Phase 4) explicitly called for scripting it so
RL credit-assignment stays on the finger contact problem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ScriptedPalmActionCfg(ActionTermCfg):
    """Config for the scripted palm action.

    Default schedule mirrors `Phase1GraspEvaluator`:
      - sim_step < settle_steps:                lift_scale = 0
      - settle_steps <= sim_step < settle+ramp: lift_scale = (sim_step - settle + 1) / ramp
      - sim_step >= settle + ramp:              lift_scale = 1
    Other palm dims (px, py, rx, ry, rz) hold at `palm_default_ctrl`.

    If `palm_rotation_residual_scale > 0`, the action term takes 3 policy
    inputs (rx, ry, rz residuals) and adds `residual_scale * action` to the
    default rotation ctrl. Used by the in-hand reorient task to give the
    policy wrist control on top of the scripted pz-lift. Default scale 0
    keeps the term backwards-compatible (action_dim = 0, palm fully scripted).
    """
    joint_names: tuple[str, ...]
    """Six palm joint names, in the order matching `palm_default_ctrl`."""
    palm_default_ctrl: tuple[float, ...]
    """Length-6 default position targets (from the keyframe key_ctrl)."""
    palm_pz_index: int = 2
    """Index into `joint_names` for palm_pz (the dimension we ramp)."""
    palm_rot_indices: tuple[int, int, int] = (3, 4, 5)
    """Indices into `joint_names` for palm rx/ry/rz."""
    settle_steps: int = 240
    lift_ramp_steps: int = 80
    lift_delta_z: float = 0.05
    palm_rotation_residual_scale: float = 0.0
    """If >0, policy outputs 3 dims (rx, ry, rz residuals); each is scaled by
    this and added to the default rotation ctrl. Use ~0.3-0.5 rad."""
    rotation_active_from_sim_step: int | None = None
    """If set, palm-rotation residuals are zeroed before this sim step
    (during settle/grasp). Default: settle_steps (grasp finishes first)."""

    def build(self, env: "ManagerBasedRlEnv") -> "ScriptedPalmAction":
        return ScriptedPalmAction(self, env)


class ScriptedPalmAction(ActionTerm):
    cfg: ScriptedPalmActionCfg

    def __init__(self, cfg: ScriptedPalmActionCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)

        joint_ids, _ = self._entity.find_joints(list(cfg.joint_names), preserve_order=True)
        self._target_ids = torch.as_tensor(joint_ids, device=self.device, dtype=torch.long)

        default = torch.as_tensor(cfg.palm_default_ctrl, device=self.device, dtype=torch.float32)
        if default.numel() != len(cfg.joint_names):
            raise ValueError(
                f"palm_default_ctrl has {default.numel()} entries; expected {len(cfg.joint_names)}"
            )
        self._default_palm = default
        self._rot_idx = torch.as_tensor(cfg.palm_rot_indices, device=self.device, dtype=torch.long)
        self._action_dim = 3 if cfg.palm_rotation_residual_scale > 0.0 else 0
        self._rot_active_from = (cfg.rotation_active_from_sim_step
                                 if cfg.rotation_active_from_sim_step is not None
                                 else int(cfg.settle_steps))

        self._sim_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if self._action_dim > 0:
            self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        dynamic_t = self._sim_step.to(torch.float32) - float(self.cfg.settle_steps)
        ramp = max(1.0, float(self.cfg.lift_ramp_steps))
        lift_scale = torch.clamp((dynamic_t + 1.0) / ramp, min=0.0, max=1.0)  # (B,)

        target = self._default_palm.unsqueeze(0).expand(self.num_envs, -1).clone()  # (B, 6)
        pz_idx = int(self.cfg.palm_pz_index)
        target[:, pz_idx] = self._default_palm[pz_idx] + lift_scale * float(self.cfg.lift_delta_z)

        if self._action_dim > 0:
            # Gate residuals to fire only after the grasp/lift phase finishes,
            # so warmstarted policies don't perturb the existing grip trajectory.
            active = (self._sim_step >= self._rot_active_from).float().unsqueeze(-1)
            scaled = self._raw_actions * float(self.cfg.palm_rotation_residual_scale) * active
            target[:, self._rot_idx] = self._default_palm[self._rot_idx].unsqueeze(0) + scaled

        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(target - encoder_bias, joint_ids=self._target_ids)

        self._sim_step += 1

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._sim_step.zero_()
            self._raw_actions.zero_()
        else:
            self._sim_step[env_ids] = 0
            self._raw_actions[env_ids] = 0.0


@dataclass(kw_only=True)
class LerpFingerActionCfg(ActionTermCfg):
    """Linear-interpolation finger action.

    The position-controller setpoint is `lerp(start_ctrl, target_ctrl, alpha)`
    where `alpha = easing(t)` and `t = clamp(sim_step / settle_sim_steps, 0, 1)`.
    The policy adds a (clipped) residual on top, so action=0 reproduces a
    pure scripted closing motion from `start_ctrl` to `target_ctrl`.

    `easing` options:
      - "linear"          : alpha = t                  (uniform close)
      - "ease_out_quad"   : alpha = 1 - (1-t)^2        (fast at start, slow at end)
      - "ease_out_cubic"  : alpha = 1 - (1-t)^3        (more aggressive ease-out)

    Why ease-out: with linear lerp the fingers approach the cube at constant
    velocity and slam into contact at the same rate they crossed the air. With
    ease-out (quad/cubic), the first ~30% of `settle_sim_steps` covers ~50% of
    the open->grip distance (fast approach, hand is clear of cube), then the
    last 70% covers the final 50% slowly — precisely the contact-formation
    window. This gives the position controller + cube physics time to settle
    instead of bouncing.
    """
    joint_names: tuple[str, ...]
    start_ctrl: tuple[float, ...]
    target_ctrl: tuple[float, ...]
    settle_sim_steps: int = 240
    residual_scale: float = 0.2
    easing: str = "linear"
    residual_active_from_sim_step: int = 0
    """Zero the policy's finger residual before this sim step — the scripted
    LerpFinger grasp runs undisturbed during the grasp/lift, then the policy
    takes over. Needed when warmstarting a reorient policy into the normal-lift
    env: a lifted-object policy is out-of-distribution on the flat-object grasp
    and its raw residuals blow up the physics (NaN). 0 = always active."""

    def build(self, env: "ManagerBasedRlEnv") -> "LerpFingerAction":
        return LerpFingerAction(self, env)


class LerpFingerAction(ActionTerm):
    cfg: LerpFingerActionCfg

    def __init__(self, cfg: LerpFingerActionCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg=cfg, env=env)

        joint_ids, _ = self._entity.find_joints(list(cfg.joint_names), preserve_order=True)
        self._target_ids = torch.as_tensor(joint_ids, device=self.device, dtype=torch.long)

        n = len(cfg.joint_names)
        if len(cfg.start_ctrl) != n or len(cfg.target_ctrl) != n:
            raise ValueError(
                f"start_ctrl ({len(cfg.start_ctrl)}) and target_ctrl ({len(cfg.target_ctrl)}) "
                f"must each match joint_names length ({n})"
            )
        self._start = torch.as_tensor(cfg.start_ctrl, device=self.device, dtype=torch.float32)
        self._target = torch.as_tensor(cfg.target_ctrl, device=self.device, dtype=torch.float32)

        self._sim_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._raw_actions = torch.zeros(self.num_envs, n, device=self.device)
        self._action_dim = n

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        denom = max(1.0, float(self.cfg.settle_sim_steps))
        t = torch.clamp(self._sim_step.to(torch.float32) / denom, 0.0, 1.0)
        easing = self.cfg.easing
        if easing == "linear":
            alpha = t
        elif easing == "ease_out_quad":
            alpha = 1.0 - (1.0 - t).pow(2)
        elif easing == "ease_out_cubic":
            alpha = 1.0 - (1.0 - t).pow(3)
        else:
            raise ValueError(f"Unknown easing '{easing}'")
        alpha = alpha.unsqueeze(-1)  # (B, 1)
        offset = (1.0 - alpha) * self._start.unsqueeze(0) + alpha * self._target.unsqueeze(0)  # (B, N)
        active = (self._sim_step >= int(self.cfg.residual_active_from_sim_step)).float().unsqueeze(-1)
        target = self._raw_actions * float(self.cfg.residual_scale) * active + offset

        encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
        self._entity.set_joint_position_target(target - encoder_bias, joint_ids=self._target_ids)

        self._sim_step += 1

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._sim_step.zero_()
            self._raw_actions.zero_()
        else:
            self._sim_step[env_ids] = 0
            self._raw_actions[env_ids] = 0.0
