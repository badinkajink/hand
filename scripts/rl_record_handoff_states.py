"""Record Policy A's ACTUAL terminal states (post-lift) into a 'handoff state
bank' for train-the-handoff. Runs A in the normal-lift env (real lift to 0.10),
and at `--record-step` captures, per env the FULL Markov state: the object root
pose (pos RELATIVE to the env origin + quat), object root velocity (REAL, 6-d),
the robot joint qpos AND qvel, plus A's last applied action (`a_last`, the value
B reads as `last_action` at the deploy seam). Policy B
then trains spawning from sampled bank states (see env_cfg handoff_state_bank),
so it learns reorient from exactly the physically-valid grips A delivers — no
synthetic/unrealistic spawn jitter.

Usage:
  uv run python scripts/rl_record_handoff_states.py \
      --policy-a results/rl/20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt \
      --num-envs 1024 --record-step 45 --output results/rl/handoff_state_bank.npz
"""
from __future__ import annotations
import argparse, dataclasses, os, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
A_OBS_DIM = 65  # Policy A: no target_axis_misalign obs term.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-a", type=Path,
                    default=ROOT / "results/rl/20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1/tensorboard/model_500.pt")
    ap.add_argument("--morphology-run", type=Path,
                    default=ROOT / "results/phase1/run18_multi_object_adapt/foundational/screwdriver_medium_flat/run_20260521_150259")
    ap.add_argument("--num-envs", type=int, default=1024)
    ap.add_argument("--record-step", type=int, default=45, help="policy step (post-lift) at which to snapshot states")
    ap.add_argument("--output", type=Path, default=ROOT / "results/rl/handoff_state_bank.npz")
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    import json
    with (args.morphology_run / "summary.json").open() as f:
        keyframe = json.load(f).get("keyframe", "open_short_manual")
    bfc = tuple(float(v) for v in np.load(args.morphology_run / "best_rollout.npz")["best_finger_ctrl"].reshape(-1))

    from morphohand.rl.env_cfg import MorphoHandEnvCfg, to_mjlab_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner
    from morphohand.rl.ppo_config import PPOConfig
    from morphohand.rl.ppo_runner import build_runner_cfg

    # 65-dim normal-lift env (real lift to 0.10), no terminations so all envs survive to record.
    env_cfg = MorphoHandEnvCfg(
        frozen_scene_xml=args.morphology_run / "frozen_scene.xml", keyframe_name=keyframe,
        foundational_run_dir=args.morphology_run, finger_default_ctrl=bfc,
        object_body_name="screwdriver_medium", num_envs=args.num_envs,
        episode_length_s=float(args.record_step) / 50.0 + 1.0,
        lift_target_z_above_init=0.10, lift_delta_z=0.10,
        finger_residual_scale=0.5, finger_close_easing="ease_out_quad",
        contact_gate_stability_rewards=True, enable_lift_terminations=False,
    )
    env = ManagerBasedRlEnv(cfg=to_mjlab_cfg(env_cfg), device="cuda:0", render_mode=None)
    wrapped = RslRlVecEnvWrapper(env)
    runner = ManipulationOnPolicyRunner(env=wrapped,
        train_cfg=dataclasses.asdict(build_runner_cfg(PPOConfig(num_envs=args.num_envs), Path("/tmp/rec"), run_name="rec")),
        log_dir="/tmp/rec/tb", device="cuda:0")
    ck = torch.load(str(args.policy_a), map_location="cpu", weights_only=False)
    runner.alg.actor.load_state_dict(ck["actor_state_dict"], strict=True); runner.alg.actor.eval()
    actor = runner.alg.actor
    obj = env.scene["cube"]; robot = env.scene["robot"]

    obs = wrapped.reset()[0]["actor"]
    a_last = None
    with torch.no_grad():
        for t in range(args.record_step):
            a = actor.mlp(obs[:, :A_OBS_DIM]) if hasattr(actor, "mlp") else actor(obs[:, :A_OBS_DIM])
            a_last = a.detach().clone()  # A's last applied action -> the seam's `last_action` obs at deploy
            obs = wrapped.step(a)[0]["actor"]
    # snapshot — FULL physical state (Markov-complete: pose + REAL velocities + grip + finger vel)
    pose_w = obj.data.root_link_pose_w.detach().clone()          # (N,7) world
    pose_rel = pose_w.clone(); pose_rel[:, :3] -= env.scene.env_origins  # pos relative to env origin
    # Object velocity: the correct mjlab attribute is `root_link_vel_w` (N,6 = lin(3)+ang(3)).
    # The old `root_link_velocity_w` does NOT exist -> the prior bank silently zeroed it, so B
    # always trained on a MOTIONLESS seam while deploy hands off a moving state.
    vel = obj.data.root_link_vel_w.detach().clone()              # (N,6) REAL world velocity
    qpos = robot.data.joint_pos.detach().clone()                 # (N, njoints)
    qvel = robot.data.joint_vel.detach().clone()                 # (N, njoints) REAL finger/palm vel
    z = pose_w[:, 2]
    # keep only envs whose object actually lifted (didn't drop) — valid handoff states
    keep = (z > (z.median() - 0.05))
    vmag = vel[keep, :3].norm(dim=1)
    print(f"[rec] object z at record step: median={float(z.median()):.3f} kept {int(keep.sum())}/{args.num_envs}")
    print(f"[rec] obj |lin_vel| at record step: mean={float(vmag.mean()):.4f} max={float(vmag.max()):.4f} m/s "
          f"(nonzero confirms the velocity-attr fix)")
    print(f"[rec] |a_last| mean={float(a_last[keep].abs().mean()):.4f} (A's delivered action -> seam last_action)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output,
             obj_pose=pose_rel[keep].cpu().numpy(), obj_vel=vel[keep].cpu().numpy(),
             robot_qpos=qpos[keep].cpu().numpy(), robot_qvel=qvel[keep].cpu().numpy(),
             a_last=a_last[keep].cpu().numpy(),
             joint_names=np.array(list(robot.joint_names)))
    print(f"[rec] saved {int(keep.sum())} handoff states (pose+vel+qpos+qvel+a_last) -> {args.output}")
    env.close()


if __name__ == "__main__":
    main()
