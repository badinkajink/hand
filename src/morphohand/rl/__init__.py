"""Reinforcement learning training pipeline for MorphoHand.

See `docs/rl/` for the design overview. The MVP targets a single (cube,
morphology) task using mjlab (MuJoCo Warp backend) with an Opt2Skill-style
tracking-augmented PPO policy, warm-started from a CEM-optimised reference
trajectory.

Modules:
- `reference_trajectory`: load + interpolate CEM rollouts as RL motion refs.
- `scene_loader`: produce frozen-scene MJCFs + index tables for the env.
- `reward`: pure-function reward terms (tracking + task + regularizers).
- `observations`: pure-function observation extractors.
- `env_cfg`: mjlab ManagerBasedRLEnvCfg assembly.
- `ppo_runner`: thin RSL-RL wrapper writing to `results/rl/<tag>/`.
- `ppo_config`: PPO hyperparameter dataclass.
"""
from __future__ import annotations
