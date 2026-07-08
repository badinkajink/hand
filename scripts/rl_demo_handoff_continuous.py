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
      --policy-b results/rl/b02_20260602-0024-policyB_v2_smooth10x_quick/tensorboard/model_1219.pt \
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
                 num_steps: int, finger_residual_scale: float = 0.5,
                 finger_close_easing: str = "ease_out_quad",
                 contact_gate_stability_rewards: bool = True,
                 lift_delta: float = 0.10, open_finger_from_keyframe: bool = False):
    """One env cfg. enable_target_axis=False -> 65-dim (Policy A's space);
    True -> 66-dim normal-lift reorient env (Policy B's space + dynamics).
    skip_lift_phase is always False here: the cylinder starts flat and is
    really lifted, so the handoff is a continuous physical rollout.

    NB the residual-scale / easing / contact-gate knobs MUST match the
    evaluated policy's TRAINING env or B is OOD (a B finetuned at scale 0.2 and
    deployed at 0.5 emits 2.5x-too-large residuals and blows up the grip)."""
    from morphohand.rl.env_cfg import MorphoHandEnvCfg
    common = dict(
        frozen_scene_xml=frozen, keyframe_name=keyframe,
        foundational_run_dir=morph, finger_default_ctrl=bfc,
        object_body_name="screwdriver_medium", num_envs=1,
        episode_length_s=float(num_steps) / 50.0 + 0.5,
        finger_residual_scale=finger_residual_scale, finger_close_easing=finger_close_easing,
        lift_target_z_above_init=lift_delta, lift_delta_z=lift_delta,
        contact_gate_stability_rewards=contact_gate_stability_rewards, enable_lift_terminations=False,
        open_finger_from_keyframe=open_finger_from_keyframe,
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


def read_contact_force(env_unwrapped, sensor_name):
    """(mean, max) contact-force magnitude in N for env 0 of a contact sensor.
    Penetration depth scales with normal force under MuJoCo soft contact, so this
    is the tractable, faithful 'how hard is it clamping' readout (the thumb-into-
    screwdriver artifact = high fingertip force). 0,0 if the sensor is missing."""
    try:
        s = env_unwrapped.scene.sensors[sensor_name]
        f = getattr(s.data, "force", None)
        if f is None:
            return 0.0, 0.0
        mag = f.norm(dim=-1, keepdim=True) if f.dim() == 2 else f.norm(dim=-1)  # (B,slots)
        m = mag[0]
        return float(m.mean()), float(m.max())
    except Exception:
        return 0.0, 0.0


def read_per_finger(env_unwrapped, sensor_name):
    """Per-finger (force_mag, found) for env 0, ordered [thumb, index, middle] (the
    fingertip_cube_contact primary body pattern). Returns ([f,f,f], [g,g,g]) or (None,None).
    The aggregate read_contact_force() means over fingers and HIDES a degenerate grip where
    one finger loses contact and the thumb carries the load."""
    try:
        s = env_unwrapped.scene.sensors[sensor_name]
        f = getattr(s.data, "force", None)
        found = getattr(s.data, "found", None)
        if f is None:
            return None, None
        mag = f.norm(dim=-1)                       # (B, n_bodies[, slots])
        if mag.dim() == 3:
            mag = mag.amax(dim=-1)
        if found is not None and found.dim() == 3:
            found = found.amax(dim=-1)
        m = mag[0].detach().cpu().numpy().tolist()
        g = (found[0].float().detach().cpu().numpy().tolist() if found is not None else [float("nan")] * len(m))
        return m, g
    except Exception:
        return None, None


def act(actor, obs):
    return actor.act_inference(obs) if hasattr(actor, "act_inference") else actor.mlp(obs)


def act_b(actor, obs_td, stochastic):
    """Policy B's action. obs_td is the full TensorDict (B is the native policy,
    so use its proper forward/normalizer path, not the raw-.mlp A shortcut)."""
    if stochastic:
        return actor(obs_td, stochastic_output=True)
    return actor(obs_td)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-a", type=Path,
                    default=ROOT / "results/rl/a01_20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt")
    ap.add_argument("--policy-b", type=Path,
                    default=ROOT / "results/rl/b02_20260602-0024-policyB_v2_smooth10x_quick/tensorboard/model_1219.pt")
    ap.add_argument("--morphology-run", type=Path,
                    default=ROOT / "results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259")
    ap.add_argument("--output", type=Path, default=ROOT / "docs/rl/videos/reorient/handoff_continuous.mp4")
    ap.add_argument("--handoff-step", type=int, default=40, help="policy step to switch A->B (after lift+settle)")
    ap.add_argument("--lift-delta", type=float, default=0.10,
                    help="scripted palm lift height (m). MUST match the evaluated A/B training "
                         "(baseline lineage=0.10; m05 native A+B trained at 0.05).")
    ap.add_argument("--open-finger-from-keyframe", action="store_true",
                    help="start fingers from the keyframe (open_ik) pose; REQUIRED for "
                         "IK-retargeted morphologies (else the LerpFinger starts from the "
                         "baseline open pose and a finger arrives late).")
    ap.add_argument("--blend-steps", type=int, default=0,
                    help="ramp A->B actions over N steps from handoff-step (0 = hard switch). "
                         "Eases B in so it is never shocked by an OOD obs at the seam.")
    ap.add_argument("--total-steps", type=int, default=240)
    ap.add_argument("--finger-residual-scale", type=float, default=0.5,
                    help="MUST match the evaluated B's training env (B10/B4 used 0.5; "
                         "the rl_train_cube default is 0.2 — a mismatch puts B OOD).")
    ap.add_argument("--finger-close-easing", type=str, default="ease_out_quad")
    ap.add_argument("--no-contact-gate", action="store_true",
                    help="set contact_gate_stability_rewards=False (match a B trained without it)")
    ap.add_argument("--action-lowpass", type=float, default=1.0,
                    help="EMA low-pass on B's actions at deploy (1.0=off; 0.5=smooth). "
                         "The NON-REWARD smoothness lever: jerk penalties backfire (the "
                         "corrective finger jerk IS the stabilization), but a deploy-time "
                         "filter removes high-freq jitter without retraining. Applied "
                         "post-handoff only; A's lift is unfiltered.")
    ap.add_argument("--stochastic-b", action="store_true",
                    help="sample Policy B's action from its distribution instead of the "
                         "deterministic mean (tests whether the corrective exploration "
                         "jitter is load-bearing for the post-seam hold)")
    # ---- 3-stage handoff: A (lift) -> B (catch+stabilize) -> C (finish reorient) ----
    # B4 reorients to 0.988 but only from a CLEAN start (drops A's raw delivery).
    # Idea: let the seam-survivor B (b32) convert A's messy delivery into a clean,
    # stable held state, THEN hand off a 2nd time to C=B4 to finish 0.78 -> 0.98.
    # Uses the survivor to MANUFACTURE the start C needs. No new training.
    ap.add_argument("--policy-c", type=Path, default=None,
                    help="3rd-stage policy (e.g. B4). If set, B catches then C finishes.")
    ap.add_argument("--handoff-step-2", type=int, default=90,
                    help="policy step to switch B->C (give B time to stabilize the catch)")
    ap.add_argument("--policy-c-residual-scale", type=float, default=0.5,
                    help="C's NATIVE finger_residual_scale (B4=0.5). C's action is rescaled "
                         "by (this / --finger-residual-scale) so its effective residual matches "
                         "its training even though the env runs at B's scale (gotcha #13).")
    ap.add_argument("--blend-steps-2", type=int, default=None,
                    help="B->C action ramp length at the 2nd seam (default = --blend-steps). "
                         "Set independently to ease C in without touching the A->B seam.")
    ap.add_argument("--diag-every", type=int, default=20,
                    help="heartbeat: print object z / lateral-drift / cos every N "
                         "post-handoff steps (0 = off). A full slip+jitter+drift "
                         "summary always prints at the end.")
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
    ap.add_argument("--record-fingertip-traj", type=Path, default=None,
                    help="also record the OBJECT-RELATIVE fingertip trajectory (post-handoff) to "
                         "this .npz — the imitation reference for morphology-transferable reorient "
                         "(src/morphohand/rl/imitation.py).")
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

    # 1) Throwaway A-env to instantiate + load Policy A's actor. Detect A's obs dim
    # from the checkpoint: a native lift Policy A is 65-dim, but a B->A CO-REFINED A
    # (warmstart-A finetuned in the 66-dim reorient env) is 66-dim. Build the A-env to
    # match, and feed A obs[:, :a_obs_dim] in the rollout.
    import torch as _torch
    _sd = _torch.load(str(args.policy_a), map_location="cpu", weights_only=False)["actor_state_dict"]
    a_obs_dim = int(next(_sd[k] for k in _sd if k.endswith("weight")).shape[1])
    print(f"[hc] building A-env ({a_obs_dim}-dim) to load Policy A...")
    env_a, _wa, actor_a = build_actor(
        make_env_cfg(frozen, keyframe, args.morphology_run, bfc, enable_target_axis=(a_obs_dim == 66), num_steps=10,
                     finger_residual_scale=args.finger_residual_scale, finger_close_easing=args.finger_close_easing, contact_gate_stability_rewards=not args.no_contact_gate, lift_delta=args.lift_delta, open_finger_from_keyframe=args.open_finger_from_keyframe),
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
    cfg_b = make_env_cfg(frozen, keyframe, args.morphology_run, bfc, enable_target_axis=True, num_steps=args.total_steps,
                         finger_residual_scale=args.finger_residual_scale, finger_close_easing=args.finger_close_easing, contact_gate_stability_rewards=not args.no_contact_gate, lift_delta=args.lift_delta, open_finger_from_keyframe=args.open_finger_from_keyframe)
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

    # 3-stage: load Policy C (e.g. B4) into its own runner. C is a native B-policy
    # (same 66-dim obs), so it consumes obs_td via act_b just like B.
    actor_c = None
    c_rescale = 1.0
    if args.policy_c is not None:
        runner_c = ManipulationOnPolicyRunner(
            env=wrapped, train_cfg=dataclasses.asdict(build_runner_cfg(PPOConfig(num_envs=1), work, run_name="hcC")),
            log_dir=str(work / "tb_tmpC"), device="cuda:0")
        ckpt_c = torch.load(str(args.policy_c), map_location="cpu", weights_only=False)
        runner_c.alg.actor.load_state_dict(ckpt_c["actor_state_dict"], strict=True)
        runner_c.alg.actor.eval()
        actor_c = runner_c.alg.actor
        c_rescale = float(args.policy_c_residual_scale) / float(args.finger_residual_scale)
        print(f"[hc] Policy C loaded (3-stage): B->C at step {args.handoff_step_2}, "
              f"C action rescale x{c_rescale:.2f} (native scale {args.policy_c_residual_scale} "
              f"/ env scale {args.finger_residual_scale}).")

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
    min_z_post = float("inf")  # min object-z AFTER the handoff (the honest hold metric;
    # whole-rollout min_z is dominated by the pre-lift floor phase z~0.012)
    held_cos_tail = []  # held-vertical cos post-handoff (reorientation quality)
    # Full per-step trajectory for the slip/jitter/drift diagnostics (the things
    # we kept having to verify by eye): (step, x, y, z, axisX, axisY, axisZ) where
    # axis is the object's body +Z (long axis) direction in world.
    traj = []
    # FULL-rollout log (from step 0) for the baked-in trajectory-health scorecard —
    # the grasp phase is needed to catch a LATE finger, which post-handoff-only logging
    # (the old diag) structurally could not see.
    full = {"found": [], "force": [], "z": [], "cos": [], "x": [], "y": [], "axis": []}
    handoff_xy = None
    ft_rec = []      # (step, (3,3)) object-relative fingertip positions, for --record-fingertip-traj
    tip_force = []   # (mean, max) fingertip-cube contact force [N], post-handoff
    palm_force = []  # max palm-cube contact force [N], post-handoff (does it ever seat?)
    finger_force = []  # per-finger [thumb,index,middle] contact force [N], post-handoff
    finger_found = []  # per-finger [thumb,index,middle] contact-found (0/1), post-handoff
    prev_action = None  # for the optional action low-pass filter
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
            # 3-stage: once past handoff_step_2, Policy C (B4) takes over from B.
            c_active = actor_c is not None and switch_step is not None and step >= args.handoff_step_2
            if switch_step is None or step < switch_step:
                actions = act(actor_a, obs[:, :a_obs_dim])
            elif args.blend_steps > 0 and step < switch_step + args.blend_steps:
                # linear A->B action ramp: alpha 0->1 over the blend window
                alpha = (step - switch_step + 1) / float(args.blend_steps)
                a_a = act(actor_a, obs[:, :a_obs_dim])
                a_b = act_b(actor_b, obs_td, args.stochastic_b)
                actions = (1.0 - alpha) * a_a + alpha * a_b
            elif c_active and (_blend2 := (args.blend_steps_2 if args.blend_steps_2 is not None else args.blend_steps)) > 0 and step < args.handoff_step_2 + _blend2:
                # linear B->C action ramp at the 2nd seam (eases B4 into B's grip)
                alpha = (step - args.handoff_step_2 + 1) / float(_blend2)
                a_b = act_b(actor_b, obs_td, args.stochastic_b)
                a_c = act_b(actor_c, obs_td, args.stochastic_b) * c_rescale
                actions = (1.0 - alpha) * a_b + alpha * a_c
            elif c_active:
                actions = act_b(actor_c, obs_td, args.stochastic_b) * c_rescale
            else:
                actions = act_b(actor_b, obs_td, args.stochastic_b)
            # NON-REWARD smoothness lever: EMA low-pass on B's deploy actions.
            if args.action_lowpass < 1.0 and switch_step is not None and step >= switch_step:
                a_lp = float(args.action_lowpass)
                actions = a_lp * actions + (1.0 - a_lp) * (
                    prev_action if prev_action is not None else actions)
            prev_action = actions
            obs_td, *_ = wrapped.step(actions)
            obs = obs_td["actor"]
            try:
                pose = env.unwrapped.scene["cube"].data.root_link_pose_w[0]
                x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
                min_z = min(min_z, z)
                # held-cos = world-z component of the object's body +Z (long axis):
                # 1 - 2(qx^2+qy^2) for quat (w,x,y,z). flat ~0, vertical ~1.
                qw, qx, qy, qz = (float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))
                held_cos = 1.0 - 2.0 * (qx * qx + qy * qy)
                # object body +Z direction in world (the long axis), for jitter/wobble.
                ax_x = 2.0 * (qx * qz + qw * qy)
                ax_y = 2.0 * (qy * qz - qw * qx)
                ax_z = held_cos
                # FULL-rollout log (every step, incl. the grasp phase) for the scorecard.
                _ff, _fg = read_per_finger(env.unwrapped, "fingertip_cube_contact")
                full["found"].append(_fg if _fg is not None else [0.0, 0.0, 0.0])
                full["force"].append(_ff if _ff is not None else [0.0, 0.0, 0.0])
                full["z"].append(z); full["cos"].append(held_cos)
                full["x"].append(x); full["y"].append(y); full["axis"].append((ax_x, ax_y, ax_z))
                if switch_step is not None and step >= switch_step:
                    min_z_post = min(min_z_post, z)
                    held_cos_tail.append(held_cos)
                    traj.append((step, x, y, z, ax_x, ax_y, ax_z))
                    if args.record_fingertip_traj is not None:
                        from morphohand.rl.imitation import fingertips_in_object_frame
                        ft_rec.append(fingertips_in_object_frame(env.unwrapped)[0].cpu().numpy())
                    tf_mean, tf_max = read_contact_force(env.unwrapped, "fingertip_cube_contact")
                    _, pf_max = read_contact_force(env.unwrapped, "palm_cube_contact")
                    tip_force.append((tf_mean, tf_max))
                    palm_force.append(pf_max)
                    pf_force, pf_found = read_per_finger(env.unwrapped, "fingertip_cube_contact")
                    if pf_force is not None:
                        finger_force.append(pf_force)
                        finger_found.append(pf_found)
                    if handoff_xy is None:
                        handoff_xy = (x, y)
                    if args.diag_every and (step - switch_step) % args.diag_every == 0:
                        lat = ((x - handoff_xy[0]) ** 2 + (y - handoff_xy[1]) ** 2) ** 0.5
                        print(f"[diag] step={step:3d}  z={z:.3f}  lat_drift={lat * 100:5.1f}cm  "
                              f"cos={held_cos:+.3f}  tipF={tf_mean:4.1f}N  palmF={pf_max:4.1f}N")
                if switch_step is not None and step == switch_step:
                    align_at_handoff = z
                if os.environ.get("HC_ZTRACE") and (step % 5 == 0 or (switch_step is not None and abs(step - switch_step) <= 6)):
                    print(f"[ztrace] step={step} z={z:.4f}")
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
    print(f"[hc] min object-center z POST-HANDOFF (honest hold metric): {min_z_post:.4f} m "
          f"({'HELD >0.05' if min_z_post > 0.05 else 'dropped'})")
    if held_cos_tail:
        tail = held_cos_tail[-50:]
        print(f"[hc] held-vertical cos POST-HANDOFF (last 50 steps mean): {sum(tail)/len(tail):.3f} "
              f"(peak {max(held_cos_tail):.3f})")

    # ---- slip / jitter / drift diagnostics (the eye-only artifacts) ----------
    # dt per policy step = decimation(10) * sim_timestep(0.002) = 0.02 s.
    if len(traj) >= 3:
        DT = 0.02
        a = np.asarray(traj, dtype=np.float64)
        pos, axis = a[:, 1:4], a[:, 4:7]
        hx, hy = handoff_xy
        lat = np.hypot(pos[:, 0] - hx, pos[:, 1] - hy)          # horiz drift from handoff
        horiz_path = float(np.abs(np.diff(pos[:, :2], axis=0)).sum())
        z_net = float(pos[-1, 2] - pos[0, 2])
        zt = pos[-min(50, len(pos)):, 2]
        z_slope = float(np.polyfit(np.arange(len(zt)), zt, 1)[0] / DT)  # m/s sink/rise
        lin_v = np.diff(pos, axis=0) / DT
        lin_speed = np.linalg.norm(lin_v, axis=1)
        lin_jerk = np.linalg.norm(np.diff(pos, n=2, axis=0) / DT ** 2, axis=1)  # |d2pos/dt2|
        dots = np.clip((axis[1:] * axis[:-1]).sum(1), -1.0, 1.0)
        ang_speed = np.arccos(dots) / DT                        # rad/s of the long axis
        ang_jerk = np.linalg.norm(np.diff(axis, n=2, axis=0) / DT ** 2, axis=1)
        print("[diag] ===== POST-HANDOFF slip / jitter / drift =====")
        print(f"[diag]  lateral drift from handoff:  mean {lat.mean()*100:5.1f}cm   "
              f"max {lat.max()*100:5.1f}cm   net {lat[-1]*100:5.1f}cm")
        print(f"[diag]  horizontal path length:      {horiz_path*100:5.1f}cm "
              f"(wander; >>net ⇒ sliding around)")
        print(f"[diag]  vertical:  net Δz {z_net*100:+5.1f}cm   tail z-slope {z_slope*100:+5.1f}cm/s "
              f"({'SINKING' if z_slope < -0.003 else 'stable'})")
        print(f"[diag]  object speed:  lin mean {lin_speed.mean()*100:5.1f} max {lin_speed.max()*100:5.1f} cm/s   "
              f"ang mean {ang_speed.mean():4.2f} max {ang_speed.max():4.2f} rad/s")
        print(f"[diag]  JITTER:  lin-jerk {lin_jerk.mean():6.2f} m/s²   "
              f"ang-jerk {ang_jerk.mean():6.2f} 1/s²  (lower = smoother)")
        tfa = np.asarray(tip_force)  # (T,2) mean,max per step
        pfa = np.asarray(palm_force)
        tip_mean = float(tfa[:, 0].mean()); tip_peak = float(tfa[:, 1].max())
        palm_mean = float(pfa.mean()); palm_peak = float(pfa.max())
        print(f"[diag]  CONTACT FORCE (penetration proxy):  fingertip mean {tip_mean:4.1f}N "
              f"peak {tip_peak:4.1f}N   palm mean {palm_mean:4.1f}N peak {palm_peak:4.1f}N")
        print(f"[diag]      (grip_force reward saturates at 3N; fingertip ≫3N ⇒ over-clamp/penetration. "
              f"palm≈0 ⇒ never seats into palm.)")
        if finger_force:
            ff = np.asarray(finger_force)   # (T,3) [thumb,index,middle]
            fg = np.asarray(finger_found)   # (T,3)
            fmean = ff.mean(0); gfrac = fg.mean(0)
            names = ("thumb", "index", "mid")
            print(f"[diag]  GRIP BALANCE (per finger):  "
                  + "  ".join(f"{names[i]} {fmean[i]:4.1f}N (touch {gfrac[i]:.2f})" for i in range(3)))
            lo = int(np.argmin(fmean)); hi = int(np.argmax(fmean))
            share = fmean[hi] / (fmean.sum() or 1.0)
            # degenerate grip = a finger barely touches AND one finger carries the load
            if gfrac.min() < 0.5 or share > 0.5:
                print(f"[diag]      ⇒ DEGENERATE GRIP: '{names[lo]}' lost (touch {gfrac[lo]:.2f}); "
                      f"'{names[hi]}' carries {100*share:.0f}% of the load. A balanced tripod would need far less force.")
        flags = []
        if lat.max() > 0.03: flags.append(f"SLIP(lat {lat.max()*100:.0f}cm)")
        if z_slope < -0.003: flags.append("SINKING")
        if ang_jerk.mean() > 30: flags.append(f"ANG-JITTER({ang_jerk.mean():.0f})")
        if lin_jerk.mean() > 2.0: flags.append(f"LIN-JITTER({lin_jerk.mean():.1f})")
        if tip_mean > 3.5: flags.append(f"OVER-CLAMP({tip_mean:.1f}N)")
        print(f"[diag]  VERDICT: {'  '.join(flags) if flags else 'clean (held firm + smooth)'}")

    # ---- BAKED-IN trajectory-health scorecard (flags LATE FINGER, drop, jitter, ------
    # idle-finger, de-centering, over-clamp — the degenerate patterns the old aggregate
    # metrics masked). Runs on EVERY handoff eval; writes JSON next to the video.
    if len(full["z"]) >= 5:
        from morphohand.rl.trajectory_health import characterize_trajectory, format_scorecard
        import json as _json
        axis = np.asarray(full["axis"])
        dots = np.clip((axis[1:] * axis[:-1]).sum(1), -1.0, 1.0)
        angvel = np.concatenate([[0.0], np.arccos(dots) / 0.02])  # rad/s from long-axis turn
        ho = switch_step if switch_step is not None else args.handoff_step
        sc = characterize_trajectory(
            finger_found=full["found"], finger_force=full["force"], obj_z=full["z"],
            obj_cos=full["cos"], obj_xy=np.stack([full["x"], full["y"]], axis=1),
            obj_angvel=angvel, grasp_end=ho, hold_start=min(ho + 15, len(full["z"]) - 1))
        print(format_scorecard(sc, title=args.output.stem))
        (args.output.with_suffix(".health.json")).write_text(_json.dumps(sc.as_dict(), indent=1))
        print(f"[health] scorecard -> {args.output.with_suffix('.health.json')}")

    # ---- optional: dump the object-relative fingertip trajectory (imitation reference) ----------
    if args.record_fingertip_traj is not None and ft_rec:
        args.record_fingertip_traj.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.record_fingertip_traj, fingertip_obj=np.stack(ft_rec).astype(np.float32),
                 dt=0.02, t0=0.0, source=str(args.policy_b))
        print(f"[imitation] object-relative fingertip trajectory ({len(ft_rec)} steps) -> "
              f"{args.record_fingertip_traj}")


if __name__ == "__main__":
    main()
