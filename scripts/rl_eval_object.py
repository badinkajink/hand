"""Deterministic eval of any object policy at full DR + pose-grid videos.

Usage:
  uv run --extra rl python scripts/rl_eval_object.py \
    --checkpoint results/rl/<tag>/tensorboard/model_<iter>.pt \
    --foundational-run results/phase1/run18_final/foundational/<object>/run_<ts> \
    --object-body-name <name> \
    --x-jitter 0.006 --y-jitter 0.010 --yaw-jitter 0.52 \
    --x-center 0.006 --y-center -0.003 \
    [--pose-grid 3x3] \
    [--num-envs 64]

Outputs (under <checkpoint dir>/../eval_<iter>/):
  metrics.json
  random_pose_eval.mp4         (64-env stochastic-pose deterministic rollout)
  pose_grid/pose_{ix}_{iy}_{iyaw}.mp4  (one per grid cell)
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
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--pose-grid", type=str, default="3x3x1",
                   help="Format NxxNyxNyaw — N evenly-spaced poses in each axis. "
                        "Total = product. Skipped if 1x1x1.")
    p.add_argument("--episode-steps", type=int, default=70)
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

    # Read the keyframe object quat so we can preserve it when overriding the
    # cube pose manually. Without this, the pose-grid videos would spawn the
    # object with identity-yawed quat — flat-laying cylinders end up standing.
    from morphohand.rl.env_cfg import _read_keyframe_object_pose
    _, base_quat = _read_keyframe_object_pose(
        RUN / "frozen_scene.xml",
        summary.get("keyframe", "open_short_manual"),
        args.object_body_name,
    )
    base_quat_t = torch.tensor(base_quat, dtype=torch.float32, device="cuda:0")
    from mjlab.utils.lab_api.math import quat_mul

    iter_num = args.checkpoint.stem.split("_")[-1]
    out_dir = args.checkpoint.parent.parent / f"eval_{iter_num}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def build_env(num_envs, video_folder, name_prefix):
        env_cfg = MorphoHandEnvCfg(
            frozen_scene_xml=RUN / "frozen_scene.xml",
            keyframe_name=summary.get("keyframe", "open_short_manual"),
            foundational_run_dir=RUN, finger_default_ctrl=bfc,
            object_body_name=args.object_body_name, num_envs=num_envs,
            cube_spawn_x_jitter=args.x_jitter, cube_spawn_y_jitter=args.y_jitter,
            cube_spawn_x_center=args.x_center, cube_spawn_y_center=args.y_center,
            cube_spawn_yaw_jitter=args.yaw_jitter,
            dr_anneal_iters=0,
            finger_residual_scale=args.finger_residual_scale,
        )
        env = ManagerBasedRlEnv(cfg=to_mjlab_cfg(env_cfg), device="cuda:0", render_mode="rgb_array")
        env = VideoRecorder(env, video_folder=video_folder,
                            step_trigger=lambda s: s == 1, video_length=args.episode_steps,
                            name_prefix=name_prefix)
        return env

    def load_actor(num_envs):
        env = build_env(num_envs, out_dir, "tmp")
        wrapped = RslRlVecEnvWrapper(env)
        ppo = PPOConfig(num_envs=num_envs)
        runner = ManipulationOnPolicyRunner(env=wrapped,
            train_cfg=dataclasses.asdict(build_runner_cfg(ppo, out_dir, run_name="eval")),
            log_dir=str(out_dir / "tb_tmp"), device="cuda:0")
        ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
        runner.alg.actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
        runner.alg.actor.eval()
        return env, wrapped, runner.alg.actor

    # --- (1) random pose deterministic rollout --------------------------
    env, wrapped, actor = load_actor(args.num_envs)
    cube = env.unwrapped.scene["cube"]
    sensor = env.unwrapped.scene.sensors["fingertip_cube_contact"]
    obs_td, _ = wrapped.reset()
    obs = obs_td["actor"]
    cz_trace, cxy_trace, cq_trace, cmin_trace = [], [], [], []
    spawn_pose = None
    with torch.no_grad():
        for t in range(args.episode_steps):
            actions = actor.act_inference(obs) if hasattr(actor, "act_inference") else actor.mlp(obs)
            obs_td, *_ = wrapped.step(actions)
            obs = obs_td["actor"]
            pose = cube.data.root_link_pose_w.clone()
            cz_trace.append(pose[:, 2].clone())
            cxy_trace.append(pose[:, :2].clone())
            cq_trace.append(pose[:, 3:7].clone())
            if t == 0:
                spawn_pose = pose.clone()
            f = sensor.data.found
            cmin_trace.append((f > 0).float().min(dim=-1).values.clone()
                              if f is not None else torch.zeros(args.num_envs, device="cuda:0"))
    cz = torch.stack(cz_trace, 0); cmin = torch.stack(cmin_trace, 0)
    cxy = torch.stack(cxy_trace, 0); cq = torch.stack(cq_trace, 0)
    peak = cz.max(0).values
    # xy drift during HOLD phase (steps 40-60): how much does the cube wander while lifted?
    # spawn_xy is post-step-0 (already inside fingers, before lift)
    xy_drift_hold = (cxy[40:60] - cxy[40:60].mean(dim=0, keepdim=True)).norm(dim=-1).mean().item()
    # orientation drift during HOLD: geodesic distance from mean hold quat
    q_hold_mean = cq[40:60].mean(dim=0)
    q_hold_mean = q_hold_mean / (q_hold_mean.norm(dim=-1, keepdim=True) + 1e-9)
    dot = (cq[40:60] * q_hold_mean.unsqueeze(0)).sum(dim=-1).abs().clamp(max=1.0)
    ori_drift_hold = (2.0 * torch.acos(dot)).mean().item()
    metrics = dict(
        median_peak_cube_z=float(torch.median(peak).item()),
        lift_success_6cm=float((peak > 0.06).float().mean().item()),
        lift_success_4cm=float((peak > 0.04).float().mean().item()),
        contact_min_hold=float(cmin[40:60].mean().item()),
        contact_min_full_episode=float(cmin.mean().item()),
        cube_xy_drift_hold_mean_m=float(xy_drift_hold),
        cube_orientation_drift_hold_mean_rad=float(ori_drift_hold),
        num_envs=args.num_envs,
        x_jitter=args.x_jitter, y_jitter=args.y_jitter, yaw_jitter=args.yaw_jitter,
        x_center=args.x_center, y_center=args.y_center,
        checkpoint=str(args.checkpoint),
    )
    print(json.dumps(metrics, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # IMPORTANT: env.close() must run BEFORE the rename — VideoRecorder
    # only flushes the mp4 to disk on close(). Renaming before close()
    # silently skips because the file doesn't exist yet, and the file
    # then gets caught by the pose-grid rename loop later.
    env.close()
    for v in out_dir.glob("tmp-step-*.mp4"):
        v.rename(out_dir / "random_pose_eval.mp4")

    # --- (2) pose-grid videos ------------------------------------------
    gx, gy, gyaw = map(int, args.pose_grid.lower().split("x"))
    if gx * gy * gyaw <= 1:
        return
    pose_dir = out_dir / "pose_grid"
    pose_dir.mkdir(exist_ok=True)

    xs = np.linspace(-args.x_jitter, args.x_jitter, gx) if gx > 1 else np.array([0.0])
    ys = np.linspace(-args.y_jitter, args.y_jitter, gy) if gy > 1 else np.array([0.0])
    yaws = np.linspace(-args.yaw_jitter, args.yaw_jitter, gyaw) if gyaw > 1 else np.array([0.0])

    total = gx * gy * gyaw
    print(f"\n[pose_grid] rendering {total} videos at fixed poses")
    env, wrapped, actor = load_actor(total)
    cube = env.unwrapped.scene["cube"]

    # Compose fixed poses (one per env, set after reset)
    poses_xy = []
    names = []
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iyaw, yaw in enumerate(yaws):
                poses_xy.append((x, y, yaw))
                names.append(f"x{ix}_y{iy}_yaw{iyaw}")

    obs_td, _ = wrapped.reset()
    # Override cube pose per env. Compose the yaw rotation with the keyframe
    # base_quat so flat-laying cylinders stay flat (their rest quat is a 90°
    # X-rotation; identity-yawed quat would flip them upright).
    init_pose = cube.data.root_link_pose_w.clone()
    pose = init_pose.clone()
    yaw_quats = torch.zeros(len(poses_xy), 4, device=pose.device, dtype=pose.dtype)
    for i, (x, y, yaw) in enumerate(poses_xy):
        pose[i, 0] = args.x_center + x + env.unwrapped.scene.env_origins[i, 0]
        pose[i, 1] = args.y_center + y + env.unwrapped.scene.env_origins[i, 1]
        cw, sw = float(np.cos(yaw / 2)), float(np.sin(yaw / 2))
        yaw_quats[i, 0] = cw; yaw_quats[i, 3] = sw
    composed = quat_mul(yaw_quats, base_quat_t.unsqueeze(0).expand_as(yaw_quats))
    pose[:, 3:7] = composed
    cube.write_root_link_pose_to_sim(pose)
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()

    obs_td = {"actor": env.unwrapped.observation_manager.compute()["actor"]}
    obs = obs_td["actor"]
    with torch.no_grad():
        for t in range(args.episode_steps):
            actions = actor.act_inference(obs) if hasattr(actor, "act_inference") else actor.mlp(obs)
            obs_td, *_ = wrapped.step(actions)
            obs = obs_td["actor"]

    # VideoRecorder only records env 0 — re-run per-env by rebuilding env with that env's pose alone.
    # We instead use this single-shot to drive env_0 through pose 0, and accept that as a
    # representative "random pose" video. For per-pose videos, we'd need num_envs single-env runs.
    env.close()

    # Single-env per-pose: this is slower but produces N labelled videos.
    print(f"[pose_grid] writing per-pose videos (single-env runs, slow):")
    for i, ((x, y, yaw), name) in enumerate(zip(poses_xy, names)):
        env, wrapped, actor = load_actor(1)
        env_inner = env.unwrapped
        cube = env_inner.scene["cube"]
        obs_td, _ = wrapped.reset()
        init_pose = cube.data.root_link_pose_w.clone()
        cw, sw = float(np.cos(yaw / 2)), float(np.sin(yaw / 2))
        init_pose[0, 0] = args.x_center + x + env_inner.scene.env_origins[0, 0]
        init_pose[0, 1] = args.y_center + y + env_inner.scene.env_origins[0, 1]
        # Compose yaw with keyframe base_quat (preserves flat orientation).
        yaw_q = torch.tensor([[cw, 0.0, 0.0, sw]], device=init_pose.device, dtype=init_pose.dtype)
        init_pose[0, 3:7] = quat_mul(yaw_q, base_quat_t.unsqueeze(0))[0]
        cube.write_root_link_pose_to_sim(init_pose)
        env_inner.scene.write_data_to_sim()
        env_inner.sim.forward()
        obs = env_inner.observation_manager.compute()["actor"]
        with torch.no_grad():
            for t in range(args.episode_steps):
                actions = actor.act_inference(obs) if hasattr(actor, "act_inference") else actor.mlp(obs)
                obs_td, *_ = wrapped.step(actions)
                obs = obs_td["actor"]
        env.close()
        # rename
        for v in out_dir.glob("tmp-step-*.mp4"):
            v.rename(pose_dir / f"{name}_x{x*1000:+.0f}mm_y{y*1000:+.0f}mm_yaw{np.degrees(yaw):+.0f}.mp4")
        print(f"  [{i+1}/{total}] {name}")

    print(f"\n[done] artifacts under {out_dir}")


if __name__ == "__main__":
    main()
