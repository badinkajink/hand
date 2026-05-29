"""Play a trained policy through N back-to-back episodes with random cube
spawns, recording one continuous video.

Each episode the LiftingCommand resamples the cube spawn pose uniformly
within the configured DR range, so you see how the policy handles a
random walk through pose space. mjlab auto-resets at episode end.

Usage:
  uv run --extra rl --extra gpu python scripts/rl_play_policy.py \
    --checkpoint results/rl/<tag>/tensorboard/model_<iter>.pt \
    --foundational-run results/phase1/run18_final/foundational/<obj>/run_<ts> \
    --object-body-name <name> \
    --x-jitter 0.02 --y-jitter 0.005 --yaw-jitter 0.52 \
    --num-episodes 10 \
    --out-dir results/rl/<tag>/play/
"""
from __future__ import annotations
import argparse, dataclasses, json, sys
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--foundational-run", type=Path, required=True)
    p.add_argument("--object-body-name", type=str, required=True)
    p.add_argument("--x-jitter", type=float, default=0.0)
    p.add_argument("--y-jitter", type=float, default=0.0)
    p.add_argument("--yaw-jitter", type=float, default=0.0)
    p.add_argument("--x-center", type=float, default=0.0)
    p.add_argument("--y-center", type=float, default=0.0)
    p.add_argument("--finger-residual-scale", type=float, default=0.5)
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--episode-steps", type=int, default=70)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    from morphohand.rl.env_cfg import MorphoHandEnvCfg, to_mjlab_cfg
    from morphohand.rl.ppo_config import PPOConfig
    from morphohand.rl.ppo_runner import build_runner_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner
    from mjlab.utils.wrappers.video_recorder import VideoRecorder

    RUN = args.foundational_run.resolve()
    with (RUN / "summary.json").open() as f:
        summary = json.load(f)
    npz = np.load(RUN / "best_rollout.npz")
    bfc = tuple(float(v) for v in np.asarray(npz["best_finger_ctrl"]).reshape(-1))

    iter_num = args.checkpoint.stem.split("_")[-1]
    out_dir = args.out_dir or args.checkpoint.parent.parent / f"play_{iter_num}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = MorphoHandEnvCfg(
        frozen_scene_xml=RUN / "frozen_scene.xml",
        keyframe_name=summary.get("keyframe", "open_short_manual"),
        foundational_run_dir=RUN, finger_default_ctrl=bfc,
        object_body_name=args.object_body_name, num_envs=1,
        cube_spawn_x_jitter=args.x_jitter, cube_spawn_y_jitter=args.y_jitter,
        cube_spawn_x_center=args.x_center, cube_spawn_y_center=args.y_center,
        cube_spawn_yaw_jitter=args.yaw_jitter,
        dr_anneal_iters=0,
        finger_residual_scale=args.finger_residual_scale,
    )
    total_frames = args.num_episodes * args.episode_steps
    env = ManagerBasedRlEnv(cfg=to_mjlab_cfg(env_cfg), device="cuda:0",
                             render_mode="rgb_array")
    env = VideoRecorder(env, video_folder=out_dir,
                         step_trigger=lambda s: s == 1,
                         video_length=total_frames,
                         name_prefix="play")

    wrapped = RslRlVecEnvWrapper(env)
    ppo = PPOConfig(num_envs=1)
    runner = ManipulationOnPolicyRunner(env=wrapped,
        train_cfg=dataclasses.asdict(build_runner_cfg(ppo, out_dir, run_name="play")),
        log_dir=str(out_dir / "tb_tmp"), device="cuda:0")
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    runner.alg.actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
    runner.alg.actor.eval()
    actor = runner.alg.actor

    cube = env.unwrapped.scene["cube"]
    sensor = env.unwrapped.scene.sensors["fingertip_cube_contact"]

    obs_td, _ = wrapped.reset()
    obs = obs_td["actor"]

    ep_summaries = []
    ep_idx = 0
    ep_spawn_xy = None
    ep_peak_z = -1e9
    ep_min_contact = 1.0
    ep_max_xy_drift = 0.0
    cur_ep_step = 0

    print(f"[play] {args.num_episodes} episodes × {args.episode_steps} steps "
          f"= {total_frames} frames -> {out_dir}/play-step-1.mp4")
    with torch.no_grad():
        for t in range(total_frames):
            actions = actor.act_inference(obs) if hasattr(actor, "act_inference") else actor.mlp(obs)
            obs_td, rew, done, info = wrapped.step(actions)
            obs = obs_td["actor"]

            pose = cube.data.root_link_pose_w[0]
            z = float(pose[2].item())
            xy = pose[:2].clone()
            if cur_ep_step == 0:
                ep_spawn_xy = xy.clone()
            if ep_spawn_xy is not None:
                drift = float((xy - ep_spawn_xy).norm().item())
                ep_max_xy_drift = max(ep_max_xy_drift, drift)
            ep_peak_z = max(ep_peak_z, z)
            f = sensor.data.found
            if f is not None:
                cmin = float((f > 0).float().min(dim=-1).values[0].item())
                ep_min_contact = min(ep_min_contact, cmin)
            cur_ep_step += 1

            # detect end-of-episode (done OR fixed-length boundary)
            episode_ended = bool(done[0].item()) or cur_ep_step >= args.episode_steps
            if episode_ended:
                ep_idx += 1
                ep_summaries.append(dict(
                    episode=ep_idx,
                    spawn_x_m=float(ep_spawn_xy[0].item()) if ep_spawn_xy is not None else None,
                    spawn_y_m=float(ep_spawn_xy[1].item()) if ep_spawn_xy is not None else None,
                    peak_z_m=ep_peak_z, min_contact=ep_min_contact,
                    max_xy_drift_m=ep_max_xy_drift,
                    succeeded=bool(ep_peak_z > 0.04),
                ))
                ep_peak_z = -1e9; ep_min_contact = 1.0
                ep_max_xy_drift = 0.0; cur_ep_step = 0; ep_spawn_xy = None

    summary_out = dict(
        checkpoint=str(args.checkpoint),
        num_episodes=ep_idx,
        success_rate_4cm=float(np.mean([1.0 if e["succeeded"] else 0.0 for e in ep_summaries])
                               if ep_summaries else 0.0),
        median_peak_z=float(np.median([e["peak_z_m"] for e in ep_summaries])
                            if ep_summaries else 0.0),
        episodes=ep_summaries,
    )
    print(json.dumps(summary_out, indent=2))
    (out_dir / "play_summary.json").write_text(json.dumps(summary_out, indent=2))
    env.close()


if __name__ == "__main__":
    main()
