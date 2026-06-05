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
    eval_video_length: int = 250
    """Number of frames per eval video clip (~5 s @ 50 Hz). Bump for
    longer-episode tasks (e.g. reorient @ 8 s = 400 policy steps)."""
    init_noise_std: float | None = None
    """Override PPO policy initial action std (lower reduces jitter)."""
    entropy_coef: float | None = None
    """Override PPO entropy bonus weight. Set to 0 to pin the policy when the
    optimal action is near-deterministic (our case: action=0 reproduces CEM)."""
    total_timesteps: int | None = None
    """Override PPOConfig.total_timesteps. e.g. 1_000_000 for a 30-iter smoke test."""
    lift_target_z_above_init: float = 0.05
    """Target lift height (m) above the settled cube position."""
    lift_delta_z: float = 0.05
    """How high the scripted palm raises during the lift ramp (m).
    Should match `lift_target_z_above_init`, since the policy doesn't
    control the wrist height directly — if delta_z < target, the policy
    gets an unreachable goal."""
    reward_mode: str = "full"
    """Reward mode: full | tracking_only."""
    obs_mode: str = "full"
    """Observation mode: full | ref_only."""
    init_actor_checkpoint: Path | None = None
    """Optional checkpoint to initialize the actor weights."""
    warmstart_critic: bool = True
    """Also warmstart the CRITIC (value function) from the checkpoint, not just
    the actor. CRITICAL for finetuning: with a fresh random critic the value
    estimates are garbage for hundreds of iters, producing bad advantages that
    push the converged actor OFF its optimum (this silently regressed every v2
    reorient finetune — held-vertical cos 0.97 -> 0.66). Same partial-load rule
    as the actor when obs dims differ."""
    warmstart_optimizer: bool = False
    """Also restore the optimizer state. Off by default (param groups must match
    exactly); the critic warmstart is the load-bearing fix."""
    freeze_actor_std: bool = False
    """Freeze the policy's std_param (requires_grad=False). Use to pin
    exploration noise when the optimal policy is near-deterministic."""
    object_body_name: str = "cube"
    """Name of the object body in the frozen scene to extract.
    'cube' for cube run; 'prism', 'screwdriver_medium', etc. for those."""
    cube_spawn_xy_jitter: float = 0.0
    """Symmetric uniform xy noise (m, ±). Used for BOTH x and y unless
    --cube-spawn-x-jitter / --cube-spawn-y-jitter override."""
    cube_spawn_x_jitter: float | None = None
    """Symmetric uniform x noise (m, ±). None = inherit cube_spawn_xy_jitter."""
    cube_spawn_y_jitter: float | None = None
    """Symmetric uniform y noise (m, ±). None = inherit cube_spawn_xy_jitter."""
    cube_spawn_x_center: float = 0.0
    """Spawn x offset (m). Use to recenter DR on the reachable region."""
    cube_spawn_y_center: float = 0.0
    """Spawn y offset (m). Use to recenter DR on the reachable region."""
    cube_spawn_yaw_jitter: float = 0.0
    """Symmetric uniform yaw noise (rad, ±). 0 = no jitter."""
    dr_anneal_iters: int = 0
    """PPO iters over which spawn jitter linearly ramps 0 → max. 0 disables curriculum."""
    tracking_anneal_iters: int = 0
    """PPO iters over which tracking-from-CEM reward weights linearly
    scale from initial → `tracking_final_scale` x initial. 0 disables.
    Pair with --dr-anneal-iters: tracking helps keep us in the CEM basin
    early; under full DR the CEM ref object_pos is the wrong signal."""
    tracking_final_scale: float = 0.0
    """Multiplier applied to tracking weights at end of anneal.
    0.0 = tracking off post-anneal."""
    object_xy_drift_weight: float = -3.0
    """Penalty weight on cube xy drift from spawn. Larger negative = more stable."""
    object_orientation_drift_weight: float = -3.0
    """Penalty weight on cube quat drift from spawn. Larger negative = more stable."""
    finger_drift_weight: float = -2.0
    """Penalty weight on finger qpos drift from grip ctrl."""
    finger_residual_scale: float = 0.2
    """Scale on policy finger residual atop the LerpFinger setpoint.
    Increase under DR so the policy can compensate for cube offset."""
    finger_close_easing: str = "linear"
    """Easing curve for the LerpFinger setpoint. linear | ease_out_quad |
    ease_out_cubic. ease_out_* = fast approach + slow contact."""
    contact_gate_stability_rewards: bool = False
    """If set, xy_drift / orientation_drift / finger_drift penalties only
    fire once >= --contact-gate-min fraction of tips touch the cube."""
    contact_gate_min: float = 0.5
    """Contact-mean threshold above which the stability gate opens."""
    enable_lift_terminations: bool = False
    """Enable early terminations during the lift hold phase
    (slip / drop / tip-loss / finger-slip). Off by default for back-compat."""
    lift_phase_start_step: int = 40
    """Policy step from which lift-phase terminations engage."""
    term_object_slip_xy: float = 0.015
    """xy drift (m) above which we terminate during lift phase."""
    term_object_slip_yaw: float = 0.5
    """Quat geodesic drift (rad) above which we terminate during lift phase."""
    term_object_drop: float = 0.02
    """Drop threshold (m below spawn) above which we terminate during lift phase."""
    term_tip_lost_steps: int = 3
    """Consecutive policy steps any tip can be off before we terminate."""
    term_finger_slip: float = 0.3
    """Finger qpos L2 drift (rad) from grip above which we terminate."""
    # ---- in-hand reorient task knobs -----------------------------------
    episode_length_s: float = 1.4
    """Episode length in seconds. Bump to 2.5+ for in-hand reorient."""
    enable_palm_rotation_residual: bool = False
    """Add 3 policy outputs for palm rx/ry/rz residuals (wrist control)."""
    palm_rotation_residual_scale: float = 0.3
    """Scale (rad) on palm rotation residuals."""
    palm_rotation_active_from_sim_step: int | None = None
    """Sim step at which palm rotation residuals turn on (default: settle_steps)."""
    enable_target_axis_reward: bool = False
    """Add target-axis alignment reward for in-hand reorient."""
    target_axis_weight: float = 0.0
    """Weight for target_axis_alignment reward (try 50)."""
    target_axis_alpha: float = 4.0
    """Sharpness of the axis-alignment reward."""
    reorient_start_step: int = 30
    """Policy step from which target_axis reward fires (after lift completes)."""
    finger_residual_active_from_step: int = 0
    """Policy step from which the policy's finger residual applies (zeroed before;
    scripted grasp runs undisturbed). Set ~reorient_start_step in normal-lift to keep
    a reorient warmstart from blowing up the flat-object grasp."""
    strict_tip_lost_termination: bool = False
    """If True, terminate immediately on any single-step tip loss in lift phase."""
    contact_min_weight: float = 30.0
    """Weight on contact_min reward. Default 30 strongly incentivizes 3-finger
    grip; drop to 10-15 for reorient tasks where occasional regrip is needed."""
    target_axis_progress_weight: float = 0.0
    """Weight on dense Δ(alignment)-per-step reward (gain shaping)."""
    target_axis_alpha_curriculum_iters: int = 0
    """If >0, anneal target_axis_alpha from start to end linearly."""
    target_axis_alpha_start: float = 0.5
    """Initial alpha (soft, wide reward basin) for the curriculum."""
    terminate_low_tilt_velocity: bool = False
    """Terminate envs not making reorient progress (kills stationary local optima)."""
    tilt_velocity_window: int = 20
    """Window (policy steps) over which to measure tilt progress."""
    tilt_velocity_min_progress: float = 0.05
    """Min Δ(alignment) over the window; below → terminate."""
    enable_floor_proximity_termination: bool = False
    """Terminate during reorient phase when object center z < object_min_z.
    Forbids floor-bracing reorient strategies (see v4 finding)."""
    object_min_z: float = 0.05
    """Min world z (m) for object center during reorient phase. For an
    8 cm cylinder, 0.05 gives ~1 cm clearance in worst-case orientation."""
    floor_proximity_phase_start_step: int | None = None
    """Policy step from which floor-proximity termination engages.
    None = same as reorient_start_step (gate kicks in after the lift)."""
    skip_lift_phase: bool = False
    """Skip the lift phase entirely (Policy B mode). Cylinder spawns
    at lifted-and-gripped pose; fingers at CEM grip; palm at lift height.
    Pair with --reorient-start-step 10 (grace for grip to settle),
    --lift-phase-start-step 10, --floor-proximity-phase-start-step 10,
    --episode-length-s 4.0, --lift-target-z-above-init 0.0."""
    skip_lift_drop_offset: float = 0.005
    """In skip-lift mode, spawn cylinder this many meters ABOVE the
    palm's lifted z so it falls into the pre-closed grip and establishes
    contact force. 5 mm is enough to settle in ~10 sim steps."""
    skip_lift_spawn_tilt_jitter: float = 0.0
    """Handoff-robustness DR: roll/pitch jitter (rad,±) on the lifted spawn. Try 0.1-0.25."""
    skip_lift_spawn_z_jitter: float = 0.0
    """Handoff-robustness DR: z jitter (m,±) on the lifted spawn height. Try 0.02-0.04."""
    handoff_dr_curriculum_iters: int = 0
    """If >0, ramp spawn tilt/z jitter 0->max over this many iters (gradual grip adaptation)."""
    handoff_state_bank: str | None = None
    """Path to a Policy-A terminal-state bank npz; spawn B from sampled states (train-the-handoff)."""
    handoff_target_bank: str | None = None
    """Branch B (un-freeze A): path to B10's initiation-set bank npz; A gets a seam-gated
    dense reward for delivering the object into B10's reorient-onset distribution."""
    handoff_target_weight: float = 0.0
    """Weight on the handoff_target_proximity reward (try ~4). 0 disables."""
    handoff_target_seam_lo: int = 33
    handoff_target_seam_hi: int = 37
    """Policy-step window the grip-proximity reward fires over (the delivery window)."""
    handoff_target_qpos_tol: float = 0.05
    """Per-joint tolerance (rad) on the finger-qpos match."""
    handoff_target_scale_mult: float = 1.0
    """Multiplier on per-joint tolerance (>1 = looser match)."""
    action_rate_weight: float = -0.005
    """Weight on the action_rate_l2 smoothness penalty. Policy B used -0.1
    (20x normal) to suppress sim-only finger jitter."""
    object_ang_acc_weight: float = 0.0
    """Weight on the object_ang_acc_l2 smoothness penalty (L2 of cylinder
    Δω/step). Policy B used -0.05. 0 disables the term."""
    object_ang_acc_phase_start_step: int = 0
    """Policy step from which the object_ang_acc penalty engages (gate so
    the initial grip settle doesn't get penalized for transient spin-up)."""
    # ---- "smooth & quick" finetune curriculum (Policy B v2) -------------
    target_axis_progress_clamp_negative: bool = False
    """If set, only positive Δ(alignment) is rewarded. Default (unset) =
    signed progress, which penalizes slipping back down (v2 slip fix)."""
    action_rate_weight_final: float | None = None
    """Final action_rate_l2 weight for the smoothness ramp (e.g. -0.5 for 5x,
    -1.0 for 10x). None = no ramp."""
    object_ang_acc_weight_final: float | None = None
    """Final object_ang_acc_l2 weight for the smoothness ramp (e.g. -0.25 for
    5x, -0.5 for 10x). None = no ramp."""
    smoothness_curriculum_start_iter: int = 200
    """PPO iter at which the smoothness ramp begins (consolidation window)."""
    smoothness_curriculum_iters: int = 0
    """Iters over which smoothness weights ramp base->final. 0 disables."""
    enable_alignment_success_termination: bool = False
    """Terminate (success) once alignment is held; rewards quickness + locks in."""
    success_align_thresh: float = 0.9
    """Alignment cos threshold for success / success-bonus."""
    success_hold_steps: int = 10
    """Consecutive aligned steps required to declare success."""
    success_bonus_weight: float = 0.0
    """Weight on the one-shot alignment_success_bonus reward. 0 disables."""
    time_cost_weight: float = 0.0
    """Weight on per-step reorient_time_cost (small negative pressures speed)."""
    speed_bonus_weight: float = 0.0
    """Weight on the one-shot alignment_speed_bonus (early-crossing). 0 disables."""
    speed_bonus_align_thresh: float = 0.9
    """Alignment cos threshold whose first crossing triggers the speed bonus."""
    # ---- de-centering penalty (Policy B v2.1) ---------------------------
    lateral_drift_weight: float = 0.0
    """Penalty on object palm-frame lateral drift from spawn (quadratic past a
    deadband). Discourages the v2 slide-sideways de-centering. Try -10 to -40."""
    lateral_drift_deadband: float = 0.01
    """Free lateral movement (m) before the penalty engages."""
    lateral_drift_power: float = 2.0
    """Exponent on (drift - deadband); 2.0 = quadratic."""
    # ---- Phase 3: bracing (palm normal force + grip strength) -----------
    brace_force_weight: float = 0.0
    """Reward for palm<->cylinder contact force, gated on alignment. 0 off. Try +5..+20."""
    brace_align_thresh: float = 0.7
    """Alignment cos at/above which the brace reward turns on."""
    brace_max_force: float = 3.0
    """Palm force (N) saturating the brace reward."""
    grip_force_weight: float = 0.0
    """Reward for normalised fingertip grip force (pinch-to-power). 0 off. Try +2..+10."""
    grip_force_max: float = 3.0
    """Fingertip force (N) saturating the grip reward."""
    grip_force_reduce: str = "mean"
    """'mean' or 'min' over the 3 fingertips."""
    brace_distance_weight: float = 0.0
    """Dense brace shaping exp(-gap/scale) pulling cylinder end to palm. 0 off. Try +5..+20."""
    brace_distance_scale: float = 0.04
    """Length scale (m) of the dense brace-distance reward."""


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

    import numpy as np
    npz = np.load(run / "best_rollout.npz")
    best_finger_ctrl = tuple(float(v) for v in np.asarray(npz["best_finger_ctrl"]).reshape(-1))
    if len(best_finger_ctrl) != 9:
        raise ValueError(f"best_finger_ctrl has {len(best_finger_ctrl)} dims; expected 9")

    # Prepend YYYYMMDD-HHMM so runs sort chronologically on disk + in
    # wandb. Can be skipped by passing a tag that already starts with
    # 8 digits + dash.
    from datetime import datetime
    if not (len(args.tag) >= 9 and args.tag[:8].isdigit() and args.tag[8] == "-"):
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        args.tag = f"{stamp}-{args.tag}"

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
        finger_default_ctrl=best_finger_ctrl,
        num_envs=args.num_envs,
        lift_target_z_above_init=args.lift_target_z_above_init,
        lift_delta_z=args.lift_delta_z,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        object_body_name=args.object_body_name,
        cube_spawn_xy_jitter=args.cube_spawn_xy_jitter,
        cube_spawn_x_jitter=args.cube_spawn_x_jitter,
        cube_spawn_y_jitter=args.cube_spawn_y_jitter,
        cube_spawn_x_center=args.cube_spawn_x_center,
        cube_spawn_y_center=args.cube_spawn_y_center,
        cube_spawn_yaw_jitter=args.cube_spawn_yaw_jitter,
        dr_anneal_iters=args.dr_anneal_iters,
        tracking_anneal_iters=args.tracking_anneal_iters,
        tracking_final_scale=args.tracking_final_scale,
        finger_residual_scale=args.finger_residual_scale,
        object_xy_drift_weight=args.object_xy_drift_weight,
        object_orientation_drift_weight=args.object_orientation_drift_weight,
        finger_drift_weight=args.finger_drift_weight,
        finger_close_easing=args.finger_close_easing,
        contact_gate_stability_rewards=args.contact_gate_stability_rewards,
        contact_gate_min=args.contact_gate_min,
        enable_lift_terminations=args.enable_lift_terminations,
        lift_phase_start_step=args.lift_phase_start_step,
        term_object_slip_xy=args.term_object_slip_xy,
        term_object_slip_yaw=args.term_object_slip_yaw,
        term_object_drop=args.term_object_drop,
        term_tip_lost_steps=args.term_tip_lost_steps,
        term_finger_slip=args.term_finger_slip,
        episode_length_s=args.episode_length_s,
        enable_palm_rotation_residual=args.enable_palm_rotation_residual,
        palm_rotation_residual_scale=args.palm_rotation_residual_scale,
        palm_rotation_active_from_sim_step=args.palm_rotation_active_from_sim_step,
        enable_target_axis_reward=args.enable_target_axis_reward,
        target_axis_weight=args.target_axis_weight,
        target_axis_alpha=args.target_axis_alpha,
        reorient_start_step=args.reorient_start_step,
        finger_residual_active_from_step=args.finger_residual_active_from_step,
        strict_tip_lost_termination=args.strict_tip_lost_termination,
        contact_min_weight=args.contact_min_weight,
        target_axis_progress_weight=args.target_axis_progress_weight,
        target_axis_alpha_curriculum_iters=args.target_axis_alpha_curriculum_iters,
        target_axis_alpha_start=args.target_axis_alpha_start,
        terminate_low_tilt_velocity=args.terminate_low_tilt_velocity,
        tilt_velocity_window=args.tilt_velocity_window,
        tilt_velocity_min_progress=args.tilt_velocity_min_progress,
        enable_floor_proximity_termination=args.enable_floor_proximity_termination,
        object_min_z=args.object_min_z,
        floor_proximity_phase_start_step=args.floor_proximity_phase_start_step,
        skip_lift_phase=args.skip_lift_phase,
        skip_lift_drop_offset=args.skip_lift_drop_offset,
        skip_lift_spawn_tilt_jitter=args.skip_lift_spawn_tilt_jitter,
        skip_lift_spawn_z_jitter=args.skip_lift_spawn_z_jitter,
        handoff_dr_curriculum_iters=args.handoff_dr_curriculum_iters,
        handoff_state_bank=args.handoff_state_bank,
        handoff_target_bank=args.handoff_target_bank,
        handoff_target_weight=args.handoff_target_weight,
        handoff_target_seam_lo=args.handoff_target_seam_lo,
        handoff_target_seam_hi=args.handoff_target_seam_hi,
        handoff_target_qpos_tol=args.handoff_target_qpos_tol,
        handoff_target_scale_mult=args.handoff_target_scale_mult,
        action_rate_weight=args.action_rate_weight,
        object_ang_acc_weight=args.object_ang_acc_weight,
        object_ang_acc_phase_start_step=args.object_ang_acc_phase_start_step,
        target_axis_progress_clamp_negative=args.target_axis_progress_clamp_negative,
        action_rate_weight_final=args.action_rate_weight_final,
        object_ang_acc_weight_final=args.object_ang_acc_weight_final,
        smoothness_curriculum_start_iter=args.smoothness_curriculum_start_iter,
        smoothness_curriculum_iters=args.smoothness_curriculum_iters,
        enable_alignment_success_termination=args.enable_alignment_success_termination,
        success_align_thresh=args.success_align_thresh,
        success_hold_steps=args.success_hold_steps,
        success_bonus_weight=args.success_bonus_weight,
        time_cost_weight=args.time_cost_weight,
        speed_bonus_weight=args.speed_bonus_weight,
        speed_bonus_align_thresh=args.speed_bonus_align_thresh,
        lateral_drift_weight=args.lateral_drift_weight,
        lateral_drift_deadband=args.lateral_drift_deadband,
        lateral_drift_power=args.lateral_drift_power,
        brace_force_weight=args.brace_force_weight,
        brace_align_thresh=args.brace_align_thresh,
        brace_max_force=args.brace_max_force,
        grip_force_weight=args.grip_force_weight,
        grip_force_max=args.grip_force_max,
        grip_force_reduce=args.grip_force_reduce,
        brace_distance_weight=args.brace_distance_weight,
        brace_distance_scale=args.brace_distance_scale,
    )
    ppo_kwargs = dict(
        num_envs=args.num_envs,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_tags=args.wandb_tags,
        upload_model=args.upload_model,
        eval_video_interval=args.eval_video_interval,
        eval_video_length=args.eval_video_length,
    )
    if args.init_noise_std is not None:
        ppo_kwargs["init_noise_std"] = args.init_noise_std
    if args.entropy_coef is not None:
        ppo_kwargs["entropy_coef"] = args.entropy_coef
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

    render_mode = "rgb_array" if args.record_videos and args.eval_video_interval > 0 else None
    print(f"[rl_train_cube] booting mjlab env ({args.num_envs} parallel) ...")
    env = ManagerBasedRlEnv(cfg=mj_env_cfg, device="cuda:0", render_mode=render_mode)

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

    if args.init_actor_checkpoint is not None:
        ckpt = torch.load(str(args.init_actor_checkpoint), map_location="cpu", weights_only=False)

        def _partial_load(module, src_state, label):
            """Copy src_state into `module`, full-copy where shapes match and
            zero-init-into-leading-slice where the target grew (new obs/action
            dims). std_param keeps its init for new dims so exploration stays
            nonzero. Returns (n_full, partial_list, skipped_list)."""
            if src_state is None:
                raise KeyError(f"checkpoint missing {label} state")
            own_state = module.state_dict()
            n_full, partial, skipped = 0, [], []
            for k, v in src_state.items():
                if k not in own_state:
                    skipped.append((k, "key not in target")); continue
                own = own_state[k]
                if own.shape == v.shape:
                    own_state[k] = v; n_full += 1
                elif own.dim() == v.dim() and all(o >= s for o, s in zip(own.shape, v.shape)):
                    new = own.clone() if "std" in k else torch.zeros_like(own)
                    new[tuple(slice(0, s) for s in v.shape)] = v
                    own_state[k] = new; partial.append((k, v.shape, own.shape))
                else:
                    skipped.append((k, f"incompatible target={tuple(own.shape)} ckpt={tuple(v.shape)}"))
            module.load_state_dict(own_state, strict=False)
            return n_full, partial, skipped

        print(f"[rl_train_cube] warmstart from {args.init_actor_checkpoint}")
        nf, partial, skipped = _partial_load(runner.alg.actor, ckpt.get("actor_state_dict"), "actor_state_dict")
        print(f"  actor: {nf} tensors fully copied")
        for k, vs, os in partial:
            print(f"  actor PARTIAL: {k}  {tuple(vs)} -> {tuple(os)}  ({'init-kept' if 'std' in k else 'zero-init'})")
        for k, reason in skipped:
            print(f"  actor SKIP: {k}  ({reason})")
        # CRITIC warmstart — the load-bearing fix: a fresh value function emits
        # garbage advantages that knock the converged actor off its optimum.
        if args.warmstart_critic and hasattr(runner.alg, "critic"):
            nf_c, partial_c, _ = _partial_load(runner.alg.critic, ckpt.get("critic_state_dict"), "critic_state_dict")
            print(f"  critic: {nf_c} tensors fully copied, {len(partial_c)} partial")
        elif args.warmstart_critic:
            print("  critic: WARN runner.alg has no .critic attribute; skipped")
        if args.warmstart_optimizer and ckpt.get("optimizer_state_dict") is not None:
            try:
                runner.alg.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                print("  optimizer: restored")
            except Exception as e:
                print(f"  optimizer: SKIP ({e})")

    if args.freeze_actor_std:
        std_param = runner.alg.actor.distribution.std_param
        std_param.requires_grad_(False)
        print(f"[rl_train_cube] froze actor std_param at {std_param.detach().cpu().numpy()}")

    if args.wandb:
        print(f"[rl_train_cube] wandb logger -> project={args.wandb_project}  "
              f"tags={args.wandb_tags}  upload_model={args.upload_model}")
    print(f"[rl_train_cube] starting PPO for {ppo_cfg.iters_for_timesteps()} iters ...")
    try:
        runner.learn(num_learning_iterations=ppo_cfg.iters_for_timesteps())
    finally:
        for obj in (wrapped, env):
            close = getattr(obj, "close", None)
            if callable(close):
                close()
    print(f"[rl_train_cube] DONE; artefacts under {out_dir}")


if __name__ == "__main__":
    main()
