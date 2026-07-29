"""Render a verification video for a skip-lift in-hand reorient policy.

Loads a finished run's `config.yaml`, rebuilds its EXACT env (skip-lift
reorient mode), loads the requested checkpoint, and rolls out one
deterministic episode while recording. This is the phase-B half of
`scripts/rl_demo_handoff.py`, but driven from the run's own serialized
config instead of hardcoded Policy B v1 args — so it faithfully renders
any v2 finetune (5x / 10x / quick) without env drift.

Usage:
  uv run python scripts/rl_render_reorient.py \
      --run results/rl/bx_20260601-2310-policyB_v2_smooth5x \
      --checkpoint model_1219.pt \
      --output docs/rl/videos/<YYYYMMDD>_reorient/<HHMM>_policyB_v2_smooth5x.mp4
"""
from __future__ import annotations
import argparse
import dataclasses
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_env_cfg_from_yaml(run_dir: Path, num_steps: int, camera: dict | None = None):
    """Reconstruct the run's MorphoHandEnvCfg from its config.yaml, with
    num_envs forced to 1 and episode length sized to the requested rollout.

    `camera` overrides only the viewer fields (size/distance/angles) — nothing
    that touches physics or reward, so the rollout stays the run's own."""
    from morphohand.rl.env_cfg import MorphoHandEnvCfg

    with (run_dir / "config.yaml").open() as f:
        cfg = yaml.safe_load(f)
    env_d = dict(cfg["env"])

    fields = {f.name for f in dataclasses.fields(MorphoHandEnvCfg)}
    kw = {k: v for k, v in env_d.items() if k in fields}
    # Path-typed fields come back as strings from YAML.
    for pk in ("frozen_scene_xml", "foundational_run_dir"):
        if kw.get(pk) is not None:
            kw[pk] = Path(kw[pk])
    # Tuple-typed fields come back as lists; runtime indexing is fine, but
    # normalise the finger ctrl / friction to tuples to match the dataclass.
    for tk in ("finger_default_ctrl", "open_finger_qpos", "object_friction",
               "target_axis_object_local", "target_axis_world"):
        if isinstance(kw.get(tk), list):
            kw[tk] = tuple(kw[tk])
    kw["num_envs"] = 1
    kw["episode_length_s"] = float(num_steps) / 50.0 + 0.5
    for k, v in (camera or {}).items():
        if v is not None:
            kw[k] = v
    return MorphoHandEnvCfg(**kw)


def load_actor(env_cfg, checkpoint: Path, video_folder: Path, name_prefix: str,
               video_length: int):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner
    from mjlab.utils.wrappers.video_recorder import VideoRecorder
    from morphohand.rl.env_cfg import to_mjlab_cfg
    from morphohand.rl.ppo_config import PPOConfig
    from morphohand.rl.ppo_runner import build_runner_cfg

    env = ManagerBasedRlEnv(
        cfg=to_mjlab_cfg(env_cfg), device="cuda:0", render_mode="rgb_array"
    )
    env = VideoRecorder(
        env, video_folder=str(video_folder),
        step_trigger=lambda s: s == 1,
        video_length=video_length,
        name_prefix=name_prefix,
    )
    wrapped = RslRlVecEnvWrapper(env)
    ppo = PPOConfig(num_envs=1)
    runner = ManipulationOnPolicyRunner(
        env=wrapped,
        train_cfg=dataclasses.asdict(build_runner_cfg(ppo, video_folder, run_name="render")),
        log_dir=str(video_folder / "tb_tmp"),
        device="cuda:0",
    )
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    runner.alg.actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
    runner.alg.actor.eval()
    return env, wrapped, runner.alg.actor


def rollout(env, wrapped, actor, num_steps: int):
    """Deterministic rollout. Also tracks min object-center z to flag any
    floor contact (a true in-hand reorient must stay clear of the ground)."""
    obs_td, _ = wrapped.reset()
    obs = obs_td["actor"]
    min_z = float("inf")
    with torch.no_grad():
        for _ in range(num_steps):
            if hasattr(actor, "act_inference"):
                actions = actor.act_inference(obs)
            else:
                actions = actor.mlp(obs)
            obs_td, *_ = wrapped.step(actions)
            obs = obs_td["actor"]
            # object root z is exposed on the underlying env's object entity
            try:
                z = float(env.unwrapped.scene["cube"].data.root_link_pose_w[0, 2])
                min_z = min(min_z, z)
            except Exception:
                pass
    env.close()
    return min_z


def find_recorded_mp4(video_folder: Path, prefix: str) -> Path:
    matches = list(video_folder.glob(f"{prefix}*.mp4"))
    if not matches:
        raise FileNotFoundError(f"no recorded video matching {prefix}* in {video_folder}")
    return max(matches, key=lambda p: p.stat().st_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run dir with config.yaml")
    ap.add_argument("--checkpoint", default="model_1219.pt",
                    help="checkpoint filename under <run>/tensorboard/")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=200, help="rollout policy steps (~4s at 50Hz)")
    # Camera-only overrides — for a diagnostic render you can actually read.
    # The in-training recorder's 320x240 shows a drop but not which pad slipped.
    ap.add_argument("--width", type=int, default=None, help="render width (run default 320)")
    ap.add_argument("--height", type=int, default=None, help="render height (run default 240)")
    ap.add_argument("--distance", type=float, default=None, help="camera distance (m)")
    ap.add_argument("--elevation", type=float, default=None, help="camera elevation (deg)")
    ap.add_argument("--azimuth", type=float, default=None, help="camera azimuth (deg)")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    ckpt = args.run / "tensorboard" / args.checkpoint
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    env_cfg = build_env_cfg_from_yaml(args.run, args.steps, camera={
        "viewer_width": args.width, "viewer_height": args.height,
        "viewer_distance": args.distance, "viewer_elevation": args.elevation,
        "viewer_azimuth": args.azimuth,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = args.output.parent / "_render_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[render] run={args.run.name} ckpt={args.checkpoint} skip_lift={env_cfg.skip_lift_phase}")
    env, wrapped, actor = load_actor(env_cfg, ckpt, work_dir, args.output.stem, args.steps)
    min_z = rollout(env, wrapped, actor, args.steps)
    mp4 = find_recorded_mp4(work_dir, args.output.stem)
    import shutil
    shutil.move(str(mp4), str(args.output))
    print(f"[render] DONE: {args.output}")
    print(f"[render] min object-center z during rollout: {min_z:.4f} m "
          f"(floor-proximity term threshold = {env_cfg.object_min_z} m)")


if __name__ == "__main__":
    main()
