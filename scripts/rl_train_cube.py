"""PPO training entry point for the cube grasp MVP.

Usage:
  uv run python scripts/rl_train_cube.py \
      --morphology-run results/phase1/run18_final/foundational/cube/run_20260521_161817 \
      --tag cube_mvp_v1

Add `--dry-run` to validate config construction and dump `config.yaml`
without launching PPO.

Defaults:
  - 1024 parallel envs (tuned for 16 GB VRAM; bump to 2048 on 24 GB+)
  - wandb logger enabled (`--no-wandb` to fall back to tensorboard)
  - video of env[0] every 50 PPO iterations -> results/rl/<tag>/eval_videos/
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import tyro

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from morphohand.rl.env_cfg import MorphoHandEnvCfg  # noqa: E402
from morphohand.rl.ppo_config import PPOConfig  # noqa: E402
from morphohand.rl.ppo_runner import (  # noqa: E402
    build_env_cfg_and_dump, build_runner_cfg, dump_runner_cfg,
)
from morphohand.rl.scene_loader import prepare_scene  # noqa: E402


@dataclasses.dataclass
class Args:
    morphology_run: Path
    """Directory containing best_rollout.npz + summary.json + frozen_scene.xml."""
    tag: str = "cube_mvp_v1"
    """Output subdir under results/rl/."""
    output_root: Path = ROOT / "results" / "rl"
    """Override output root (default: results/rl/)."""
    dry_run: bool = False
    """Validate construction + dump config only; do not launch PPO."""
    num_envs: int = 1024
    """Parallel envs (default 1024 for 16 GB VRAM; try 512 if OOM)."""
    seed: int = 42
    wandb: bool = True
    """Sync to wandb. Use --no-wandb to log to tensorboard only."""
    wandb_project: str = "morphohand-rl"
    wandb_tags: tuple[str, ...] = ()
    """Comma-separated tags for the wandb run."""
    upload_model: bool = False
    """Upload checkpoints to wandb on save (disk-heavy; default off)."""
    record_videos: bool = True
    """Record env[0] rollout videos under results/rl/<tag>/eval_videos/."""
    eval_video_interval: int = 50
    """PPO iterations between eval video recordings."""
    total_timesteps: int | None = None
    """Override PPOConfig.total_timesteps. e.g. 1_000_000 for a 30-iter smoke test."""


def main() -> None:
    args = tyro.cli(Args)

    run = Path(args.morphology_run).resolve()
    if not (run / "best_rollout.npz").exists():
        raise FileNotFoundError(f"missing best_rollout.npz under {run}")
    if not (run / "summary.json").exists():
        raise FileNotFoundError(f"missing summary.json under {run}")
    frozen = run / "frozen_scene.xml"
    if not frozen.exists():
        # Re-freeze from summary's base scene as a fallback.
        with (run / "summary.json").open() as f:
            summary = json.load(f)
        prepare_scene(
            base_scene_xml=Path(summary["scene_xml"]),
            keyframe=summary["keyframe"],
            output_dir=run,
            object_body_name="cube",
        )
        frozen = run / f"frozen_{Path(summary['scene_xml']).stem}.xml"

    with (run / "summary.json").open() as f:
        summary = json.load(f)
    keyframe = summary.get("keyframe", "open_short_manual")

    out_dir = args.output_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "eval_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rl_train_cube] tag={args.tag}  out_dir={out_dir}")
    print(f"[rl_train_cube] frozen_scene={frozen}  keyframe={keyframe}")

    env_cfg = MorphoHandEnvCfg(
        frozen_scene_xml=frozen,
        keyframe_name=keyframe,
        foundational_run_dir=run,
        num_envs=args.num_envs,
    )
    ppo_kwargs = dict(
        num_envs=args.num_envs,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        upload_model=args.upload_model,
        eval_video_interval=args.eval_video_interval,
    )
    if args.total_timesteps is not None:
        ppo_kwargs["total_timesteps"] = args.total_timesteps
    ppo_cfg = PPOConfig(**ppo_kwargs)

    print(f"[rl_train_cube] building mjlab env cfg ...")
    mj_env_cfg = build_env_cfg_and_dump(env_cfg, ppo_cfg, out_dir)
    runner_cfg = build_runner_cfg(ppo_cfg, out_dir, run_name=args.tag)
    dump_runner_cfg(runner_cfg, out_dir)
    print(f"[rl_train_cube] dumped config.yaml + rsl_rl_cfg.json")

    if args.dry_run:
        print(f"[rl_train_cube] --dry-run set; exiting without training.")
        return

    # ---- launch PPO via mjlab's runner --------------------------------
    try:
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.utils.wrappers.video_recorder import VideoRecorder
    except ImportError as e:
        raise RuntimeError(
            f"RL extra not installed: {e}\n"
            "Install with: uv sync --extra gpu --extra rl"
        ) from e

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — PPO training requires GPU.")

    print(f"[rl_train_cube] booting mjlab env ({args.num_envs} parallel) ...")
    env = ManagerBasedRlEnv(cfg=mj_env_cfg, device="cuda:0")

    # Wrap in VideoRecorder before RslRlVecEnvWrapper so frame capture
    # sees the raw step() calls. step_trigger fires once per PPO iteration
    # boundary (every num_envs * num_steps_per_env total env steps), so a
    # video per `eval_video_interval` iterations.
    if args.record_videos and args.eval_video_interval > 0:
        step_period = args.eval_video_interval * ppo_cfg.num_steps_per_env
        print(f"[rl_train_cube] video recording every {step_period} env steps "
              f"(~ {args.eval_video_interval} PPO iters)")
        env = VideoRecorder(
            env,
            video_folder=video_dir,
            step_trigger=lambda s, p=step_period: s > 0 and (s % p) == 0,
            video_length=ppo_cfg.eval_video_length,
            name_prefix=args.tag,
        )

    wrapped = RslRlVecEnvWrapper(env)

    train_cfg = dataclasses.asdict(runner_cfg)
    runner = ManipulationOnPolicyRunner(
        env=wrapped,
        train_cfg=train_cfg,
        log_dir=str(log_dir),
        device="cuda:0",
    )

    if args.wandb:
        print(f"[rl_train_cube] wandb logger -> project={args.wandb_project}  "
              f"tags={args.wandb_tags}  upload_model={args.upload_model}")
    print(f"[rl_train_cube] starting PPO for {ppo_cfg.iters_for_timesteps()} iters ...")
    runner.learn(num_learning_iterations=ppo_cfg.iters_for_timesteps())
    print(f"[rl_train_cube] DONE; artefacts under {out_dir}")


if __name__ == "__main__":
    main()
