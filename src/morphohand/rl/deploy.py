"""Deploy-time env/actor building + contact readouts for trained A/B policies.

Extracted (CODEBASE_AUDIT.md step 2) from scripts/rl_demo_handoff_continuous.py so
policy_healthcheck.py and future eval tools import a library instead of another script.

TRAIN/DEPLOY PARITY (gotcha #13): the residual-scale / easing / contact-gate knobs of
`make_env_cfg` MUST match the evaluated policy's TRAINING env or the policy is OOD — a B
finetuned at finger_residual_scale 0.2 and deployed at 0.5 emits 2.5x-too-large residuals
and blows up the grip. The recipe layer (audit step 4) pins these in one place.

Heavy imports (mjlab, torch env build) stay inside the functions so importing this module
is cheap and CUDA-free.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import torch

A_OBS_DIM = 65  # Policy A trained without the target_axis_misalign obs term.


def ckpt_obs_dim(checkpoint: Path) -> int:
    """Actor input width straight from the checkpoint: 65 = Policy A (lift), 66 = a
    reorienter (adds the target_axis obs). Lets eval tools build a matching env."""
    sd = torch.load(str(checkpoint), map_location="cpu", weights_only=False)["actor_state_dict"]
    return int(next(sd[k] for k in sd if k.endswith("weight")).shape[1])


def make_env_cfg(frozen, keyframe, morph, bfc, *, enable_target_axis: bool,
                 num_steps: int, finger_residual_scale: float = 0.5,
                 finger_close_easing: str = "ease_out_quad",
                 contact_gate_stability_rewards: bool = True,
                 lift_delta: float = 0.10, open_finger_from_keyframe: bool = False):
    """One env cfg. enable_target_axis=False -> 65-dim (Policy A's space);
    True -> 66-dim normal-lift reorient env (Policy B's space + dynamics).
    skip_lift_phase is always False here: the cylinder starts flat and is
    really lifted, so a handoff is a continuous physical rollout."""
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
    """Policy A's action from the raw actor obs tensor (sliced to A's obs dim)."""
    return actor.act_inference(obs) if hasattr(actor, "act_inference") else actor.mlp(obs)


def act_b(actor, obs_td, stochastic):
    """Policy B's action. obs_td is the full TensorDict (B is the native policy,
    so use its proper forward/normalizer path, not the raw-.mlp A shortcut)."""
    if stochastic:
        return actor(obs_td, stochastic_output=True)
    return actor(obs_td)
