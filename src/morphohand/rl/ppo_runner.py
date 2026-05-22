"""Thin wrapper around mjlab's RSL-RL runner.

Translates `MorphoHandEnvCfg` + `PPOConfig` into a live PPO training loop,
writing artefacts to `results/rl/<tag>/`. The runner already handles
checkpointing, tensorboard/wandb logging, video recording, and ONNX export.

This module uses mjlab's `RslRlOnPolicyRunnerCfg` dataclass as the source
of truth for the runner config (so wandb_project / upload_model / etc.
stay in sync with whatever mjlab expects).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .env_cfg import MorphoHandEnvCfg, to_mjlab_cfg
from .ppo_config import PPOConfig


def build_env_cfg_and_dump(env_cfg: MorphoHandEnvCfg, ppo_cfg: PPOConfig,
                            output_dir: Path) -> Any:
    """Build the mjlab env cfg, override num_envs from PPOConfig, dump to YAML."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mj_cfg = to_mjlab_cfg(env_cfg)
    mj_cfg.scene.num_envs = ppo_cfg.num_envs
    snapshot = {
        "env": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(env_cfg).items()},
        "ppo": asdict(ppo_cfg),
    }
    with (output_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False)
    return mj_cfg


def build_runner_cfg(ppo_cfg: PPOConfig, output_dir: Path, run_name: str):
    """Build mjlab's `RslRlOnPolicyRunnerCfg` from our PPOConfig.

    Returns the structured dataclass; convert via `dataclasses.asdict()`
    when handing to `MjlabOnPolicyRunner` (which takes a plain dict).
    """
    from mjlab.rl.config import (
        RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg,
    )

    actor = RslRlModelCfg(
        hidden_dims=ppo_cfg.actor_hidden_dims,
        activation=ppo_cfg.activation,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": ppo_cfg.init_noise_std,
            "std_type": "scalar",
        },
    )
    critic = RslRlModelCfg(
        hidden_dims=ppo_cfg.critic_hidden_dims,
        activation=ppo_cfg.activation,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=ppo_cfg.num_learning_epochs,
        num_mini_batches=ppo_cfg.num_mini_batches,
        learning_rate=ppo_cfg.learning_rate,
        schedule=ppo_cfg.schedule,
        gamma=ppo_cfg.gamma,
        lam=ppo_cfg.lam,
        entropy_coef=ppo_cfg.entropy_coef,
        desired_kl=ppo_cfg.desired_kl,
        max_grad_norm=ppo_cfg.max_grad_norm,
        value_loss_coef=ppo_cfg.value_loss_coef,
        use_clipped_value_loss=ppo_cfg.use_clipped_value_loss,
        clip_param=ppo_cfg.clip_param,
    )
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=ppo_cfg.num_steps_per_env,
        max_iterations=ppo_cfg.iters_for_timesteps(),
        save_interval=ppo_cfg.save_interval,
        experiment_name=Path(output_dir).name,
        run_name=run_name,
        logger="wandb" if ppo_cfg.wandb_enabled else "tensorboard",
        wandb_project=ppo_cfg.wandb_project,
        wandb_tags=tuple(ppo_cfg.wandb_tags),
        upload_model=ppo_cfg.upload_model,
        actor=actor,
        critic=critic,
        algorithm=algorithm,
    )


def dump_runner_cfg(runner_cfg, output_dir: Path) -> None:
    """JSON dump the runner cfg for reproducibility."""
    with (Path(output_dir) / "rsl_rl_cfg.json").open("w") as f:
        json.dump(dataclasses.asdict(runner_cfg), f, indent=2, default=str)
