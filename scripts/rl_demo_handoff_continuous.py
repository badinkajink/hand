"""State-CONTINUOUS Policy A -> Policy B handoff in a single environment.

Unlike rl_demo_handoff.py (two separate envs + ffmpeg concat, which shows a
visual "teleport" at the switch), this rolls out ONE env with NO reset between
policies: Policy A's actions drive the lift, then at the handoff step Policy B's
actions take over the SAME physics state.

How the obs-dim mismatch is handled (no retraining needed):
  Policy A was trained with enable_target_axis_reward=False  -> 65-dim obs.
  Policy B was trained with it True (adds `target_axis_misalign`, appended LAST)
  -> 66-dim obs. We build the single env in B-mode (66-dim) and feed Policy A the
  first 65 dims (obs[:, :65]); the base-65 terms are identical between the two
  envs, so A sees exactly its training observation.

Lift height: the env lifts to 0.10 m (matching Policy B's skip-lift spawn height)
so B starts in-distribution; Policy A (trained at 0.05) only has to hold the grip
through the scripted ramp, which it tolerates. Any residual jump at the handoff is
a REAL train/deploy distribution gap (the thing to close by training B's successor
in this normal-lift env).

Usage:
  uv run python scripts/rl_demo_handoff_continuous.py \
      --policy-b results/rl/20260602-0024-policyB_v2_smooth10x_quick/tensorboard/model_1219.pt \
      --output docs/rl/videos/reorient/handoff_continuous.mp4
"""
from __future__ import annotations
import argparse
import dataclasses
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

A_OBS_DIM = 65  # Policy A trained without the target_axis_misalign obs term.


def make_env_cfg(frozen, keyframe, morph, bfc, *, enable_target_axis: bool,
                 num_steps: int):
    """One env cfg. enable_target_axis=False -> 65-dim (Policy A's space);
    True -> 66-dim normal-lift reorient env (Policy B's space + dynamics).
    skip_lift_phase is always False here: the cylinder starts flat and is
    really lifted, so the handoff is a continuous physical rollout."""
    from morphohand.rl.env_cfg import MorphoHandEnvCfg
    common = dict(
        frozen_scene_xml=frozen, keyframe_name=keyframe,
        foundational_run_dir=morph, finger_default_ctrl=bfc,
        object_body_name="screwdriver_medium", num_envs=1,
        episode_length_s=float(num_steps) / 50.0 + 0.5,
        finger_residual_scale=0.5, finger_close_easing="ease_out_quad",
        lift_target_z_above_init=0.10, lift_delta_z=0.10,
        contact_gate_stability_rewards=True, enable_lift_terminations=False,
    )
    if not enable_target_axis:
        return MorphoHandEnvCfg(**common)
    return MorphoHandEnvCfg(
        **common,
        enable_target_axis_reward=True, target_axis_weight=100.0,
        target_axis_alpha=4.0, reorient_start_step=10,
        target_axis_progress_weight=300.0,
    )


def build_actor(env_cfg, checkpoint: Path, work_dir: Path):
    """Build an env from env_cfg, instantiate the runner's actor sized to that
    env, load the checkpoint, return (env, wrapped, actor)."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner
    from morphohand.rl.env_cfg import to_mjlab_cfg
    from morphohand.rl.ppo_config import PPOConfig
    from morphohand.rl.ppo_runner import build_runner_cfg

    env = ManagerBasedRlEnv(cfg=to_mjlab_cfg(env_cfg), device="cuda:0", render_mode=None)
    wrapped = RslRlVecEnvWrapper(env)
    runner = ManipulationOnPolicyRunner(
        env=wrapped,
        train_cfg=dataclasses.asdict(build_runner_cfg(PPOConfig(num_envs=1), work_dir, run_name="hc")),
        log_dir=str(work_dir / "tb_tmp"), device="cuda:0")
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    runner.alg.actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
    runner.alg.actor.eval()
    return env, wrapped, runner.alg.actor


def act(actor, obs):
    return actor.act_inference(obs) if hasattr(actor, "act_inference") else actor.mlp(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-a", type=Path,
                    default=ROOT / "results/rl/20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt")
    ap.add_argument("--policy-b", type=Path,
                    default=ROOT / "results/rl/20260602-0024-policyB_v2_smooth10x_quick/tensorboard/model_1219.pt")
    ap.add_argument("--morphology-run", type=Path,
                    default=ROOT / "results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259")
    ap.add_argument("--output", type=Path, default=ROOT / "docs/rl/videos/reorient/handoff_continuous.mp4")
    ap.add_argument("--handoff-step", type=int, default=40, help="policy step to switch A->B (after lift+settle)")
    ap.add_argument("--blend-steps", type=int, default=0,
                    help="ramp A->B actions over N steps from handoff-step (0 = hard switch). "
                         "Eases B in so it is never shocked by an OOD obs at the seam.")
    ap.add_argument("--total-steps", type=int, default=240)
    # ---- Branch D: critic-gated switch -------------------------------------
    # Instead of a human picking --handoff-step, let B's value function choose
    # the moment to take over: switch when V_B(obs) peaks during A's delivery
    # (B is most confident it can succeed from here). Removes the arbitrary
    # clock as a source of seam variance.
    ap.add_argument("--switch-on-critic", action="store_true",
                    help="pick the A->B switch step online from B's critic value peak")
    ap.add_argument("--min-switch-step", type=int, default=25,
                    help="earliest step the critic gate may switch (let the lift settle)")
    ap.add_argument("--max-switch-step", type=int, default=90,
                    help="hard cap: switch here even if no clear V_B peak")
    ap.add_argument("--critic-patience", type=int, default=5,
                    help="switch after V_B fails to beat its running max for this many steps")
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import json
    import numpy as np
    with (args.morphology_run / "summary.json").open() as f:
        keyframe = json.load(f).get("keyframe", "open_short_manual")
    bfc = tuple(float(v) for v in np.load(args.morphology_run / "best_rollout.npz")["best_finger_ctrl"].reshape(-1))
    frozen = args.morphology_run / "frozen_scene.xml"
    work = args.output.parent / "_hc_tmp"
    work.mkdir(parents=True, exist_ok=True)

    # 1) Throwaway 65-dim A-env just to instantiate + load Policy A's actor.
    print("[hc] building A-env (65-dim) to load Policy A...")
    env_a, _wa, actor_a = build_actor(
        make_env_cfg(frozen, keyframe, args.morphology_run, bfc, enable_target_axis=False, num_steps=10),
        args.policy_a, work)
    env_a.close()
    print("[hc] Policy A loaded.")

    # 2) Main 66-dim normal-lift env + Policy B, with video recording.
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner
    from mjlab.utils.wrappers.video_recorder import VideoRecorder
    from morphohand.rl.env_cfg import to_mjlab_cfg
    from morphohand.rl.ppo_config import PPOConfig
    from morphohand.rl.ppo_runner import build_runner_cfg

    print("[hc] building main env (66-dim, normal lift to 0.10) + Policy B...")
    cfg_b = make_env_cfg(frozen, keyframe, args.morphology_run, bfc, enable_target_axis=True, num_steps=args.total_steps)
    env = ManagerBasedRlEnv(cfg=to_mjlab_cfg(cfg_b), device="cuda:0", render_mode="rgb_array")
    env = VideoRecorder(env, video_folder=str(work), step_trigger=lambda s: s == 1,
                        video_length=args.total_steps, name_prefix=args.output.stem)
    wrapped = RslRlVecEnvWrapper(env)
    runner_b = ManipulationOnPolicyRunner(
        env=wrapped, train_cfg=dataclasses.asdict(build_runner_cfg(PPOConfig(num_envs=1), work, run_name="hcB")),
        log_dir=str(work / "tb_tmpB"), device="cuda:0")
    ckpt = torch.load(str(args.policy_b), map_location="cpu", weights_only=False)
    runner_b.alg.actor.load_state_dict(ckpt["actor_state_dict"], strict=True)
    runner_b.alg.actor.eval()
    actor_b = runner_b.alg.actor

    # Branch D: B's critic (value function), used only to time the switch.
    critic_b = None
    if args.switch_on_critic:
        critic_b = runner_b.alg.critic
        critic_b.load_state_dict(ckpt["critic_state_dict"], strict=True)
        critic_b.eval()

    def value_b(obs_dict):
        # rsl_rl critic indexes the obs by group (e.g. obs["critic"]), so it
        # needs the full TensorDict, not the raw actor tensor.
        if hasattr(critic_b, "evaluate"):
            return float(critic_b.evaluate(obs_dict).reshape(-1)[0])
        return float(critic_b(obs_dict).reshape(-1)[0])

    # 3) Single continuous rollout, NO reset at the handoff.
    print(f"[hc] rolling out: A for steps 0..{args.handoff_step}, then B (no reset)...")
    obs_td, _ = wrapped.reset()
    obs = obs_td["actor"]
    align_at_handoff = None
    min_z = float("inf")
    # Resolved switch step: fixed clock, or chosen online by the critic gate.
    switch_step = None if args.switch_on_critic else args.handoff_step
    vmax, t_star, no_improve, vtrace = float("-inf"), None, 0, []
    with torch.no_grad():
        for step in range(args.total_steps):
            # ---- Branch D: decide the switch step from B's value peak --------
            if args.switch_on_critic and switch_step is None:
                v = value_b(obs_td)
                vtrace.append((step, v))
                if v > vmax:
                    vmax, t_star, no_improve = v, step, 0
                elif step >= args.min_switch_step:
                    no_improve += 1
                if step >= args.min_switch_step and (
                        no_improve >= args.critic_patience or step >= args.max_switch_step):
                    switch_step = step
                    print(f"[hc] critic-gated switch at step {step} "
                          f"(V_B peak at step {t_star}, V={vmax:.3f})")

            # ---- action selection -------------------------------------------
            if switch_step is None or step < switch_step:
                actions = act(actor_a, obs[:, :A_OBS_DIM])
            elif args.blend_steps > 0 and step < switch_step + args.blend_steps:
                # linear A->B action ramp: alpha 0->1 over the blend window
                alpha = (step - switch_step + 1) / float(args.blend_steps)
                a_a = act(actor_a, obs[:, :A_OBS_DIM])
                a_b = act(actor_b, obs)
                actions = (1.0 - alpha) * a_a + alpha * a_b
            else:
                actions = act(actor_b, obs)
            obs_td, *_ = wrapped.step(actions)
            obs = obs_td["actor"]
            try:
                z = float(env.unwrapped.scene["cube"].data.root_link_pose_w[0, 2])
                min_z = min(min_z, z)
                if switch_step is not None and step == switch_step:
                    align_at_handoff = z
            except Exception:
                pass
    env.close()
    if args.switch_on_critic and vtrace:
        print("[hc] V_B trajectory (step:value): "
              + " ".join(f"{s}:{v:.2f}" for s, v in vtrace))

    # 4) Move the recorded clip to the output.
    import shutil
    matches = sorted(work.glob(f"{args.output.stem}*.mp4"), key=lambda p: p.stat().st_size)
    if not matches:
        raise FileNotFoundError(f"no recorded video in {work}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(matches[-1]), str(args.output))
    print(f"[hc] DONE: {args.output}")
    print(f"[hc] object z at handoff step {switch_step}: {align_at_handoff}")
    print(f"[hc] min object-center z over rollout: {min_z:.4f} m")


if __name__ == "__main__":
    main()
