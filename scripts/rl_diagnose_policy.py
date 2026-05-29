"""Diagnose whether a trained policy actually adapts its actions to cube pose.

The question: are the policy's residuals pose-dependent (good — closed-loop)
or pose-invariant (bad — it's just replaying CEM open-loop with token noise)?

Procedure:
  1. Build env with `num_envs == grid_size**2 * num_yaws` (default 9 envs).
  2. Override cube spawn pose per env to span a (x, y, yaw) grid.
  3. Roll out the deterministic policy.
  4. Per timestep, compute:
       - mean abs(residual)        — how big are residuals?
       - cross-env std(residual)   — do they differ between envs?
       - mean per-finger ctrl after residual + offset
  5. Plot all three vs time; save metrics + per-finger residual trajectories.

A pose-adapting policy has cross-env std significantly above the noise floor
(~1e-3) in the contact phase (steps 8-32). A pose-invariant policy hugs zero.

Usage:
  uv run --extra rl --extra gpu python scripts/rl_diagnose_policy.py \
    --checkpoint results/rl/<tag>/tensorboard/model_<iter>.pt \
    --foundational-run results/phase1/run18_final/foundational/cube/run_<ts> \
    --object-body-name cube \
    --x-jitter 0.02 --y-jitter 0.005 --yaw-jitter 0.52 \
    --grid 3x3x1
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
    p.add_argument("--grid", type=str, default="3x3x1",
                   help="GxxGyxGyaw — total = product")
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

    gx, gy, gyaw = map(int, args.grid.lower().split("x"))
    n_envs = gx * gy * gyaw
    if n_envs < 2:
        raise ValueError("grid must give >= 2 envs for cross-env variance")

    RUN = args.foundational_run.resolve()
    with (RUN / "summary.json").open() as f:
        summary = json.load(f)
    npz = np.load(RUN / "best_rollout.npz")
    bfc = tuple(float(v) for v in np.asarray(npz["best_finger_ctrl"]).reshape(-1))

    iter_num = args.checkpoint.stem.split("_")[-1]
    out_dir = args.out_dir or args.checkpoint.parent.parent / f"diagnose_{iter_num}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = MorphoHandEnvCfg(
        frozen_scene_xml=RUN / "frozen_scene.xml",
        keyframe_name=summary.get("keyframe", "open_short_manual"),
        foundational_run_dir=RUN, finger_default_ctrl=bfc,
        object_body_name=args.object_body_name, num_envs=n_envs,
        cube_spawn_x_jitter=0.0, cube_spawn_y_jitter=0.0,
        cube_spawn_x_center=args.x_center, cube_spawn_y_center=args.y_center,
        cube_spawn_yaw_jitter=0.0,
        dr_anneal_iters=0,
        finger_residual_scale=args.finger_residual_scale,
    )
    env = ManagerBasedRlEnv(cfg=to_mjlab_cfg(env_cfg), device="cuda:0",
                             render_mode=None)
    wrapped = RslRlVecEnvWrapper(env)
    ppo = PPOConfig(num_envs=n_envs)
    runner = ManipulationOnPolicyRunner(env=wrapped,
        train_cfg=dataclasses.asdict(build_runner_cfg(ppo, out_dir, run_name="diagnose")),
        log_dir=str(out_dir / "tb_tmp"), device="cuda:0")
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    runner.alg.actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
    runner.alg.actor.eval()
    actor = runner.alg.actor

    cube = env.scene["cube"]

    # Build the (x, y, yaw) grid (each axis 1 -> 0; >=2 -> linspace)
    xs = np.linspace(-args.x_jitter, args.x_jitter, gx) if gx > 1 else np.array([0.0])
    ys = np.linspace(-args.y_jitter, args.y_jitter, gy) if gy > 1 else np.array([0.0])
    yaws = np.linspace(-args.yaw_jitter, args.yaw_jitter, gyaw) if gyaw > 1 else np.array([0.0])
    poses = []
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            for iyaw, yaw in enumerate(yaws):
                poses.append((float(x), float(y), float(yaw)))

    # Reset, override cube pose per env, then forward sim
    obs_td, _ = wrapped.reset()
    init_pose = cube.data.root_link_pose_w.clone()
    pose = init_pose.clone()
    for i, (x, y, yaw) in enumerate(poses):
        pose[i, 0] = args.x_center + x + env.scene.env_origins[i, 0]
        pose[i, 1] = args.y_center + y + env.scene.env_origins[i, 1]
        cw, sw = float(np.cos(yaw / 2)), float(np.sin(yaw / 2))
        pose[i, 3] = cw; pose[i, 4] = 0.0; pose[i, 5] = 0.0; pose[i, 6] = sw
    cube.write_root_link_pose_to_sim(pose)
    env.scene.write_data_to_sim()
    env.sim.forward()
    obs = env.observation_manager.compute()["actor"]

    # Roll out and record actions
    action_trace = []  # (T, n_envs, 9)
    with torch.no_grad():
        for t in range(args.episode_steps):
            actions = actor.act_inference(obs) if hasattr(actor, "act_inference") else actor.mlp(obs)
            action_trace.append(actions.clone())
            obs_td, *_ = wrapped.step(actions)
            obs = obs_td["actor"]
    a = torch.stack(action_trace, 0).cpu().numpy()  # (T, n_envs, 9)
    T, N, D = a.shape

    # Per-timestep diagnostics
    abs_mean = np.abs(a).mean(axis=(1, 2))          # (T,) — mean |residual| across envs/fingers
    abs_per_env = np.abs(a).mean(axis=2)            # (T, n_envs)
    cross_env_std = a.std(axis=1).mean(axis=1)      # (T,) — variance across envs (per finger then mean)
    # Per-finger version
    cross_env_std_per_finger = a.std(axis=1)        # (T, 9)

    finger_names = [
        "thumb_yaw", "thumb_mcp", "thumb_pip",
        "index_yaw", "index_mcp", "index_pip",
        "middle_yaw", "middle_mcp", "middle_pip",
    ]
    contact_window = slice(8, 32)  # closing + lift ramp
    metrics = dict(
        checkpoint=str(args.checkpoint),
        grid=args.grid,
        num_envs=N, episode_steps=T,
        poses=[dict(x=x, y=y, yaw=yaw) for x, y, yaw in poses],
        # peak residual magnitude
        max_abs_residual=float(np.abs(a).max()),
        mean_abs_residual_full_episode=float(abs_mean.mean()),
        mean_abs_residual_contact_window=float(abs_mean[contact_window].mean()),
        # cross-env variance (the key diagnostic)
        mean_cross_env_std_full_episode=float(cross_env_std.mean()),
        mean_cross_env_std_contact_window=float(cross_env_std[contact_window].mean()),
        per_finger_cross_env_std_contact_window={
            name: float(cross_env_std_per_finger[contact_window, i].mean())
            for i, name in enumerate(finger_names)
        },
    )
    # Verdict
    if metrics["mean_cross_env_std_contact_window"] < 5e-3:
        verdict = ("POSE-INVARIANT: cross-env std < 5e-3 in contact window. "
                   "Policy ignores cube-pose obs and replays CEM open-loop.")
    elif metrics["mean_cross_env_std_contact_window"] < 5e-2:
        verdict = ("WEAKLY POSE-ADAPTIVE: cross-env std 5e-3 .. 5e-2. "
                   "Policy varies slightly with pose but not aggressively.")
    else:
        verdict = ("POSE-ADAPTIVE: cross-env std > 5e-2 in contact window. "
                   "Policy meaningfully adjusts residuals to cube pose.")
    metrics["verdict"] = verdict
    print(json.dumps(metrics, indent=2))
    (out_dir / "diagnose.json").write_text(json.dumps(metrics, indent=2))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        t_axis = np.arange(T)
        axes[0].plot(t_axis, abs_mean, label="mean |residual| (all envs)")
        for i in range(N):
            x, y, yaw = poses[i]
            axes[0].plot(t_axis, abs_per_env[:, i], alpha=0.3,
                         label=f"env{i} x={x*1000:+.0f} y={y*1000:+.0f} yaw={np.degrees(yaw):+.0f}")
        axes[0].set_ylabel("|residual|")
        axes[0].set_title(f"Residual magnitude — {metrics['verdict']}")
        axes[0].legend(fontsize=7, ncol=2)
        axes[0].axvspan(8, 32, alpha=0.1, color="orange", label="contact window")

        axes[1].plot(t_axis, cross_env_std, label="mean over fingers")
        for i, name in enumerate(finger_names):
            axes[1].plot(t_axis, cross_env_std_per_finger[:, i], alpha=0.5, label=name)
        axes[1].set_ylabel("cross-env std")
        axes[1].set_title("Cross-env std (high = pose-adaptive)")
        axes[1].axhline(5e-3, color="red", linestyle=":", label="invariant threshold")
        axes[1].axhline(5e-2, color="green", linestyle=":", label="adaptive threshold")
        axes[1].axvspan(8, 32, alpha=0.1, color="orange")
        axes[1].legend(fontsize=6, ncol=3)
        axes[1].set_yscale("log")

        # Per-finger residual trajectories, one subplot per finger, all envs overlaid
        axes[2].axvspan(8, 32, alpha=0.1, color="orange")
        for i, name in enumerate(finger_names):
            mean_traj = a[:, :, i].mean(axis=1)
            axes[2].plot(t_axis, mean_traj, label=name, alpha=0.8)
        axes[2].set_ylabel("mean residual per finger")
        axes[2].set_xlabel("policy step")
        axes[2].set_title("Per-finger residual trajectory (mean over envs)")
        axes[2].legend(fontsize=7, ncol=3)

        fig.tight_layout()
        png = out_dir / "diagnose.png"
        fig.savefig(png, dpi=110)
        print(f"\n[diagnose] plot -> {png}")
    except Exception as e:
        print(f"[diagnose] plot skipped: {e}")

    env.close()


if __name__ == "__main__":
    main()
