"""Single-process demo: deploy Policy A (lift) and Policy B (reorient)
in sequence, hand off control, write one video.

Two sequential rollouts in the same process — Policy A's normal-mode env
runs until lift completes; then Policy B's skip-lift-mode env runs the
reorient phase. The two clips are concatenated with ffmpeg.

This is NOT a state-continuous handoff: Policy A's terminal physics state
isn't transferred into Policy B's env (their envs differ — Policy A used
lift_delta_z=0.05, Policy B used 0.10 + skip-lift spawn). State continuity
would require either training in a shared env mode or hand-built state
injection across env builds; this demo prioritizes shippable output.

Usage:
  uv run python scripts/rl_demo_handoff.py
"""
from __future__ import annotations
import argparse
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_env(env_cfg, video_folder: Path, name_prefix: str, video_length: int):
    """Build a single-env ManagerBasedRlEnv with VideoRecorder attached."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.wrappers.video_recorder import VideoRecorder
    from morphohand.rl.env_cfg import to_mjlab_cfg

    env = ManagerBasedRlEnv(
        cfg=to_mjlab_cfg(env_cfg), device="cuda:0", render_mode="rgb_array"
    )
    env = VideoRecorder(
        env, video_folder=str(video_folder),
        step_trigger=lambda s: s == 1,
        video_length=video_length,
        name_prefix=name_prefix,
    )
    return env


def load_actor(env_cfg, checkpoint: Path, video_folder: Path, name_prefix: str,
                video_length: int):
    """Build env, wrap, instantiate runner, load checkpoint weights."""
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner
    from morphohand.rl.ppo_config import PPOConfig
    from morphohand.rl.ppo_runner import build_runner_cfg

    env = build_env(env_cfg, video_folder, name_prefix, video_length)
    wrapped = RslRlVecEnvWrapper(env)
    ppo = PPOConfig(num_envs=1)
    runner = ManipulationOnPolicyRunner(
        env=wrapped,
        train_cfg=dataclasses.asdict(build_runner_cfg(ppo, video_folder, run_name="demo")),
        log_dir=str(video_folder / "tb_tmp"),
        device="cuda:0",
    )
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    runner.alg.actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
    runner.alg.actor.eval()
    return env, wrapped, runner.alg.actor


def rollout(env, wrapped, actor, num_steps: int):
    """Deterministic rollout for `num_steps`. VideoRecorder captures frames."""
    obs_td, _ = wrapped.reset()
    obs = obs_td["actor"]
    with torch.no_grad():
        for _ in range(num_steps):
            if hasattr(actor, "act_inference"):
                actions = actor.act_inference(obs)
            else:
                actions = actor.mlp(obs)
            obs_td, *_ = wrapped.step(actions)
            obs = obs_td["actor"]
    env.close()


def find_recorded_mp4(video_folder: Path, prefix: str) -> Path:
    matches = list(video_folder.glob(f"{prefix}*.mp4"))
    if not matches:
        raise FileNotFoundError(f"no recorded video matching {prefix}* in {video_folder}")
    return max(matches, key=lambda p: p.stat().st_size)


def ffmpeg_concat_with_labels(phase_a_mp4: Path, phase_b_mp4: Path, output: Path):
    """Stack labels on each clip, concatenate with a brief gap frame."""
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(phase_a_mp4),
        "-i", str(phase_b_mp4),
        "-filter_complex",
        (
            f"[0:v]drawtext=fontfile={font}:text='Policy A: lift':"
            "x=4:y=4:fontsize=14:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=3[a];"
            f"[1:v]drawtext=fontfile={font}:text='Policy B: reorient':"
            "x=4:y=4:fontsize=14:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=3[b];"
            "[a][b]concat=n=2:v=1:a=0[v]"
        ),
        "-map", "[v]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-a", type=Path,
                    default=ROOT / "results/rl/20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt")
    ap.add_argument("--policy-b", type=Path,
                    default=ROOT / "results/rl/20260601-1033-policyB_v1/tensorboard/model_2000.pt")
    ap.add_argument("--morphology-run", type=Path,
                    default=ROOT / "results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "docs/rl/videos/reorient/handoff_demo.mp4")
    ap.add_argument("--phase-a-steps", type=int, default=80,
                    help="Policy steps for phase A (lift). ~1.6s at 50Hz.")
    ap.add_argument("--phase-b-steps", type=int, default=200,
                    help="Policy steps for phase B (reorient). ~4s at 50Hz.")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    # Read CEM grip for the morphology
    import json
    import numpy as np
    with (args.morphology_run / "summary.json").open() as f:
        summary = json.load(f)
    npz = np.load(args.morphology_run / "best_rollout.npz")
    bfc = tuple(float(v) for v in np.asarray(npz["best_finger_ctrl"]).reshape(-1))
    keyframe = summary.get("keyframe", "open_short_manual")
    frozen = args.morphology_run / "frozen_scene.xml"

    from morphohand.rl.env_cfg import MorphoHandEnvCfg

    work_dir = args.output.parent / "_handoff_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase A: medium_flat_stable_v1 in normal lift env -------------
    print("[handoff] Phase A: building env (normal lift mode)...")
    env_a_cfg = MorphoHandEnvCfg(
        frozen_scene_xml=frozen, keyframe_name=keyframe,
        foundational_run_dir=args.morphology_run,
        finger_default_ctrl=bfc,
        object_body_name="screwdriver_medium",
        num_envs=1,
        episode_length_s=float(args.phase_a_steps) / 50.0 + 0.5,
        lift_target_z_above_init=0.05,
        finger_residual_scale=0.5,
        finger_close_easing="ease_out_quad",
        tracking_anneal_iters=200,
        object_orientation_drift_weight=-3.0,
        finger_drift_weight=-2.0,
        contact_gate_stability_rewards=True,
        enable_lift_terminations=True,
        # Defaults from medium_flat_stable_v1 config.yaml
    )
    print("[handoff] Phase A: loading Policy A and rolling out...")
    env_a, wrap_a, actor_a = load_actor(env_a_cfg, args.policy_a, work_dir,
                                         "phaseA", args.phase_a_steps)
    rollout(env_a, wrap_a, actor_a, args.phase_a_steps)
    phaseA_mp4 = find_recorded_mp4(work_dir, "phaseA")
    print(f"[handoff] Phase A done → {phaseA_mp4.name}")

    # --- Phase B: policyB_v1 in skip-lift reorient env -----------------
    print("[handoff] Phase B: building env (skip-lift reorient mode)...")
    env_b_cfg = MorphoHandEnvCfg(
        frozen_scene_xml=frozen, keyframe_name=keyframe,
        foundational_run_dir=args.morphology_run,
        finger_default_ctrl=bfc,
        object_body_name="screwdriver_medium",
        num_envs=1,
        episode_length_s=float(args.phase_b_steps) / 50.0 + 0.5,
        lift_target_z_above_init=0.0,
        lift_delta_z=0.10,
        finger_residual_scale=0.5,
        finger_close_easing="ease_out_quad",
        tracking_anneal_iters=0,
        object_orientation_drift_weight=0.0,
        finger_drift_weight=-0.3,
        contact_gate_stability_rewards=True,
        enable_lift_terminations=True,
        lift_phase_start_step=10,
        term_object_slip_xy=0.5,
        term_object_slip_yaw=10.0,
        term_tip_lost_steps=10,
        term_finger_slip=100.0,
        enable_target_axis_reward=True,
        target_axis_weight=100.0,
        target_axis_alpha=4.0,
        reorient_start_step=10,
        contact_min_weight=15.0,
        target_axis_progress_weight=300.0,
        target_axis_alpha_curriculum_iters=0,
        target_axis_alpha_start=4.0,
        enable_floor_proximity_termination=True,
        object_min_z=0.05,
        floor_proximity_phase_start_step=10,
        skip_lift_phase=True,
    )
    print("[handoff] Phase B: loading Policy B and rolling out...")
    env_b, wrap_b, actor_b = load_actor(env_b_cfg, args.policy_b, work_dir,
                                         "phaseB", args.phase_b_steps)
    rollout(env_b, wrap_b, actor_b, args.phase_b_steps)
    phaseB_mp4 = find_recorded_mp4(work_dir, "phaseB")
    print(f"[handoff] Phase B done → {phaseB_mp4.name}")

    # --- Concat ----------------------------------------------------------
    print(f"[handoff] Concatenating → {args.output}")
    ffmpeg_concat_with_labels(phaseA_mp4, phaseB_mp4, args.output)
    print(f"[handoff] DONE: {args.output}")


if __name__ == "__main__":
    main()
