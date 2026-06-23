"""Record Policy B10's INITIATION SET — the object-state distribution B10 sits
in at reorient-onset, where it reorients to held-cos ~0.977.

Branch B (un-freeze Policy A): we want to fine-tune Policy A to DELIVER the object
into states B10 can actually reorient from. This script rebuilds B10's EXACT env
from its `config.yaml` (normal lift to 0.10, residual@35, reorient@50), rolls B10
out deterministically to `--record-step` (the handoff step, default 40), and snapshots
per-env the object root pose (pos RELATIVE to env origin + quat) and velocity, plus
robot joint qpos. `mjlab_terms.handoff_target_proximity` then rewards A for matching
the per-dim MEAN of this object-state distribution over the seam window.

Usage:
  uv run --extra rl --extra gpu python scripts/rl_record_initiation_bank.py \
      --policy-b results/rl/b10_20260604-1642-policyB_holdonlyws_repro:model_541.pt \
      --num-envs 512 --record-step 40 \
      --output results/rl/b10_initiation_bank.npz
"""
from __future__ import annotations
import argparse, dataclasses, os, sys
from pathlib import Path
import numpy as np, torch, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-b", type=str,
                    default="results/rl/b10_20260604-1642-policyB_holdonlyws_repro:model_541.pt",
                    help="run_dir:model_N.pt for B10 (its config.yaml defines the env)")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--record-step", type=int, default=40,
                    help="policy step to snapshot (the handoff step B takes over at)")
    ap.add_argument("--output", type=Path, default=ROOT / "results/rl/b10_initiation_bank.npz")
    args = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    run_dir, ckpt_name = args.policy_b.split(":")
    run_dir = (ROOT / run_dir).resolve() if not Path(run_dir).is_absolute() else Path(run_dir)

    from morphohand.rl.env_cfg import MorphoHandEnvCfg, to_mjlab_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner
    from morphohand.rl.ppo_config import PPOConfig
    from morphohand.rl.ppo_runner import build_runner_cfg

    # Rebuild B10's EXACT env from its config.yaml (same pattern as rl_eval_reorient_metrics).
    with (run_dir / "config.yaml").open() as f:
        env_d = dict(yaml.safe_load(f)["env"])
    fields = {fl.name for fl in dataclasses.fields(MorphoHandEnvCfg)}
    kw = {k: v for k, v in env_d.items() if k in fields}
    for pk in ("frozen_scene_xml", "foundational_run_dir"):
        if kw.get(pk):
            kw[pk] = Path(kw[pk])
    for tk in ("finger_default_ctrl", "open_finger_qpos", "object_friction",
               "target_axis_object_local", "target_axis_world"):
        if isinstance(kw.get(tk), list):
            kw[tk] = tuple(kw[tk])
    kw["num_envs"] = args.num_envs
    kw["episode_length_s"] = args.record_step / 50.0 + 1.0
    env = ManagerBasedRlEnv(cfg=to_mjlab_cfg(MorphoHandEnvCfg(**kw)), device="cuda:0", render_mode=None)
    wrapped = RslRlVecEnvWrapper(env)
    runner = ManipulationOnPolicyRunner(env=wrapped,
        train_cfg=dataclasses.asdict(build_runner_cfg(PPOConfig(num_envs=args.num_envs), Path("/tmp/recinit"), run_name="ri")),
        log_dir="/tmp/recinit/tb", device="cuda:0")
    ck = torch.load(str(run_dir / "tensorboard" / ckpt_name), map_location="cpu", weights_only=False)
    runner.alg.actor.load_state_dict(ck["actor_state_dict"], strict=True)
    runner.alg.actor.eval()
    actor = runner.alg.actor
    obj = env.scene["cube"]; robot = env.scene["robot"]

    obs = wrapped.reset()[0]["actor"]
    with torch.no_grad():
        for _ in range(args.record_step):
            a = actor.mlp(obs) if hasattr(actor, "mlp") else actor(obs)
            obs = wrapped.step(a)[0]["actor"]
    # snapshot at reorient-onset / handoff step
    pose_w = obj.data.root_link_pose_w.detach().clone()                  # (N,7) world
    pose_rel = pose_w.clone(); pose_rel[:, :3] -= env.scene.env_origins   # pos rel env origin
    vel = (obj.data.root_link_velocity_w.detach().clone()
           if hasattr(obj.data, "root_link_velocity_w")
           else torch.zeros(args.num_envs, 6, device="cuda:0"))
    qpos = robot.data.joint_pos.detach().clone()                         # (N, njoints)
    z = pose_w[:, 2]
    # keep only envs that are still holding (didn't drop) — valid initiation states
    keep = (z > (z.median() - 0.05))
    print(f"[recinit] object z at record step {args.record_step}: "
          f"median={float(z.median()):.3f} mean={float(z.mean()):.3f} "
          f"kept {int(keep.sum())}/{args.num_envs}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output,
             obj_pose=pose_rel[keep].cpu().numpy(), obj_vel=vel[keep].cpu().numpy(),
             robot_qpos=qpos[keep].cpu().numpy(),
             joint_names=np.array(list(robot.joint_names)),
             record_step=np.int64(args.record_step))
    print(f"[recinit] saved {int(keep.sum())} B10 initiation states -> {args.output}")
    env.close()


if __name__ == "__main__":
    main()
