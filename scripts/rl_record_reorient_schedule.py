"""Record a reorienter's held-cos-vs-time SCHEDULE from its clean spawn.

This is the TEACHER trajectory for on-policy schedule-tracking (train b29 to
meet-or-exceed B3's reorientation curve through the live-A seam, without
imitating B3's raw actions — which are OOD on b29's A-delivered grip).

Rolls out the policy deterministically from its OWN (clean, centered) spawn and
saves the per-step mean held-cos from `reorient_start_step` onward:
  schedule[k] = mean_envs cos at (reorient_start + k).
B3 rises ~0 -> ~0.99 over the window; that rising curve is the target.

Usage:
  uv run python scripts/rl_record_reorient_schedule.py \
      --run results/rl/b03_20260602-1636-policyB_abl_signed \
      --checkpoint model_405.pt \
      --output results/rl/b3_reorient_schedule.npz --steps 150 --num-envs 64
"""
from __future__ import annotations
import argparse, dataclasses, sys
from pathlib import Path
import numpy as np, torch, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=150, help="schedule length from reorient_start")
    ap.add_argument("--num-envs", type=int, default=64)
    args = ap.parse_args()

    from morphohand.rl.env_cfg import MorphoHandEnvCfg, to_mjlab_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner
    from morphohand.rl.ppo_config import PPOConfig
    from morphohand.rl.ppo_runner import build_runner_cfg

    with (args.run / "config.yaml").open() as f:
        env_d = dict(yaml.safe_load(f)["env"])
    fields = {fl.name for fl in dataclasses.fields(MorphoHandEnvCfg)}
    kw = {k: v for k, v in env_d.items() if k in fields}
    for pk in ("frozen_scene_xml", "foundational_run_dir"):
        if kw.get(pk): kw[pk] = Path(kw[pk])
    for tk in ("finger_default_ctrl", "open_finger_qpos", "object_friction",
               "target_axis_object_local", "target_axis_world"):
        if isinstance(kw.get(tk), list): kw[tk] = tuple(kw[tk])
    reorient_start = int(kw.get("reorient_start_step", 10))
    N, T = args.num_envs, reorient_start + args.steps + 2
    kw["num_envs"] = N; kw["episode_length_s"] = T / 50.0 + 0.5
    env = ManagerBasedRlEnv(cfg=to_mjlab_cfg(MorphoHandEnvCfg(**kw)), device="cuda:0", render_mode=None)
    wrapped = RslRlVecEnvWrapper(env)
    runner = ManipulationOnPolicyRunner(env=wrapped,
        train_cfg=dataclasses.asdict(build_runner_cfg(PPOConfig(num_envs=N), Path("/tmp/sched"), run_name="s")),
        log_dir="/tmp/sched/tb", device="cuda:0")
    ck = torch.load(str(args.run / "tensorboard" / args.checkpoint), map_location="cpu", weights_only=False)
    runner.alg.actor.load_state_dict(ck["actor_state_dict"], strict=True); runner.alg.actor.eval()
    actor = runner.alg.actor; obj = env.scene["cube"]
    obs_td, _ = wrapped.reset(); obs = obs_td["actor"]
    cos_hist = []
    with torch.no_grad():
        for _ in range(T):
            a = actor.mlp(obs) if hasattr(actor, "mlp") else actor(obs)
            obs_td, *_ = wrapped.step(a); obs = obs_td["actor"]
            q = obj.data.root_link_pose_w[:, 3:7]; qw, qx, qy, qz = q.unbind(-1)
            cos_hist.append((1 - 2 * (qx * qx + qy * qy)).clamp(-1, 1).mean().item())
    # schedule[k] = mean cos at k steps AFTER the reorienter takes the flat object
    # (= step 0 here; B3 spawns flat). Aligns to b29's handoff onset, when b29
    # likewise receives the flat object. Monotone-max -> non-decreasing aspirational
    # curve ("best reached by step k").
    traj = np.array(cos_hist[0:args.steps], dtype=np.float32)
    sched = np.maximum.accumulate(traj)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, schedule=sched, raw=traj, reorient_start=reorient_start,
             source=str(args.run / "tensorboard" / args.checkpoint))
    env.close()
    print(f"[sched] saved {args.output}  len={len(sched)}  "
          f"cos {sched[0]:.3f} -> {sched[-1]:.3f} (peak {traj.max():.3f})")


if __name__ == "__main__":
    main()
