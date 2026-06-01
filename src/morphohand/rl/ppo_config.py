"""PPO hyperparameters for the morphohand cube MVP.

Mirrors `rsl_rl.modules.PpoConfig` / `OnPolicyRunner` knobs but as a plain
dataclass we can store alongside the env config. Values match Phase 5 of
the plan; tune `num_envs` down on <16 GB VRAM.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PPOConfig:
    # rollout
    num_envs: int = 1024            # default tuned for RTX 4070 16GB; bump to 2048 on 24GB+
    num_steps_per_env: int = 24     # 24 * 1024 ≈ 25k transitions / update
    total_timesteps: int = 200_000_000

    # network
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    init_noise_std: float = 0.3

    # PPO
    learning_rate: float = 5e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 5e-3
    value_loss_coef: float = 1.0
    desired_kl: float = 0.01
    use_clipped_value_loss: bool = True
    schedule: str = "adaptive"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    max_grad_norm: float = 1.0

    # logging
    save_interval: int = 50         # checkpoints / N iterations
    eval_video_interval: int = 50   # record env[0] rollout video / N iterations
    eval_video_length: int = 250    # max frames per recording (~5 s @ 50 Hz)
    log_dir_root: str = "results/rl"

    # wandb
    wandb_enabled: bool = True
    wandb_project: str = "morphohand-rl"
    wandb_tags: tuple[str, ...] = ()
    upload_model: bool = True       # push checkpoints to wandb on save

    def iters_for_timesteps(self) -> int:
        return max(1, self.total_timesteps // (self.num_envs * self.num_steps_per_env))
