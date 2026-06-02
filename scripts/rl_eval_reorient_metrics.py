"""Deterministic behavioral metrics for in-hand reorient policies — the HONEST
evaluation (use this, NOT training reward sums, which conflate how-long x how-aligned
and silently hid the v2 verticality regression).

For each (run_dir, checkpoint) it rebuilds the run's exact env from config.yaml, rolls
out N envs deterministically for T steps, and reports:
  held_cos  : mean (body +Z . world +Z) over the last 50 steps — how well it HOLDS vertical
  peak_cos  : best alignment reached
  obj_jerk  : mean |Δω|^2 per step — object angular jerk (smoothness; lower = smoother)
  min_z     : min object-center height (floor clearance; < object_min_z ⇒ floor contact)
  drop      : fraction of envs whose object fell below 0.05 m

Usage:
  uv run python scripts/rl_eval_reorient_metrics.py \
      v1=results/rl/20260601-1033-policyB_v1:model_2033.pt \
      signed=results/rl/<dir>:model_405.pt
"""
from __future__ import annotations
import sys, dataclasses
from pathlib import Path
import numpy as np, torch, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run_one(run_dir: Path, ckpt_name: str, N=32, T=200):
    from morphohand.rl.env_cfg import MorphoHandEnvCfg, to_mjlab_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner
    from morphohand.rl.ppo_config import PPOConfig
    from morphohand.rl.ppo_runner import build_runner_cfg

    with (run_dir / "config.yaml").open() as f:
        env_d = dict(yaml.safe_load(f)["env"])
    fields = {fl.name for fl in dataclasses.fields(MorphoHandEnvCfg)}
    kw = {k: v for k, v in env_d.items() if k in fields}
    for pk in ("frozen_scene_xml", "foundational_run_dir"):
        if kw.get(pk): kw[pk] = Path(kw[pk])
    for tk in ("finger_default_ctrl", "open_finger_qpos", "object_friction",
               "target_axis_object_local", "target_axis_world"):
        if isinstance(kw.get(tk), list): kw[tk] = tuple(kw[tk])
    kw["num_envs"] = N; kw["episode_length_s"] = T / 50.0 + 0.5
    env = ManagerBasedRlEnv(cfg=to_mjlab_cfg(MorphoHandEnvCfg(**kw)), device="cuda:0", render_mode=None)
    wrapped = RslRlVecEnvWrapper(env)
    runner = ManipulationOnPolicyRunner(env=wrapped,
        train_cfg=dataclasses.asdict(build_runner_cfg(PPOConfig(num_envs=N), Path("/tmp/evalm"), run_name="e")),
        log_dir="/tmp/evalm/tb", device="cuda:0")
    ck = torch.load(str(run_dir / "tensorboard" / ckpt_name), map_location="cpu", weights_only=False)
    runner.alg.actor.load_state_dict(ck["actor_state_dict"], strict=True); runner.alg.actor.eval()
    actor = runner.alg.actor; obj = env.scene["cube"]
    obs_td, _ = wrapped.reset(); obs = obs_td["actor"]
    cos_hist = []; minz = torch.full((N,), 9.9, device="cuda:0"); prev_w = None; jerk = torch.zeros(N, device="cuda:0"); jn = 0
    with torch.no_grad():
        for _ in range(T):
            a = actor.mlp(obs) if hasattr(actor, "mlp") else actor(obs)
            obs_td, *_ = wrapped.step(a); obs = obs_td["actor"]
            q = obj.data.root_link_pose_w[:, 3:7]; qw, qx, qy, qz = q.unbind(-1)
            cos_hist.append((1 - 2 * (qx * qx + qy * qy)).clamp(-1, 1))
            minz = torch.minimum(minz, obj.data.root_link_pose_w[:, 2])
            w = obj.data.root_link_ang_vel_w
            if prev_w is not None: jerk = jerk + ((w - prev_w) ** 2).sum(-1); jn += 1
            prev_w = w.clone()
    C = torch.stack(cos_hist, 0)
    out = (C[-50:].mean().item(), C.max(0).values.mean().item(),
           float((jerk / max(1, jn)).mean()), float(minz.mean()), (minz < 0.05).float().mean().item())
    env.close()
    return out


def main():
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    specs = sys.argv[1:]
    if not specs:
        print(__doc__); return
    print(f"{'policy':20s} {'held_cos':>9s} {'peak':>7s} {'obj_jerk':>9s} {'min_z':>7s} {'drop':>6s}")
    for spec in specs:
        name, rest = spec.split("=", 1)
        run, ck = rest.split(":")
        try:
            h, p, jk, mz, dr = run_one(Path(run), ck)
            print(f"{name:20s} {h:9.3f} {p:7.3f} {jk:9.4f} {mz:7.3f} {dr:6.2f}")
        except Exception as e:
            print(f"{name:20s} ERROR {e}")


if __name__ == "__main__":
    main()
