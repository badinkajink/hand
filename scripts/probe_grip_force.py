"""Probe: in a policy's OWN standalone reorient env (clean held reset), measure the
fingertip AND palm contact force + held-cos. Settles the decisive question behind the
grip-quality stall: is B3's gentleness a SEATED grip (palm force > 0) or a gentle
FINGERTIP grip (palm ~ 0, just lower force than the live-A lineage's ~6.6 N)?

Reuses rl_eval_reorient_metrics' exact env rebuild from config.yaml.

Usage:
  uv run python scripts/probe_grip_force.py \
    B3=results/rl/b03_20260602-1636-policyB_abl_signed:model_405.pt \
    B4=results/rl/b04_20260603-1746-policyB_p2_lateral_only:model_541.pt
"""
from __future__ import annotations
import sys, dataclasses, os
from pathlib import Path
import numpy as np, torch, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def read_force(env_unwrapped, sensor_name):
    try:
        s = env_unwrapped.scene.sensors[sensor_name]
        f = getattr(s.data, "force", None)
        if f is None:
            return None
        mag = f.norm(dim=-1) if f.dim() == 2 else f.norm(dim=-1)  # (B,slots) or (B,)
        return mag  # tensor on device, per-env (mean over slots done by caller)
    except Exception:
        return None


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
        train_cfg=dataclasses.asdict(build_runner_cfg(PPOConfig(num_envs=N), Path("/tmp/probe"), run_name="p")),
        log_dir="/tmp/probe/tb", device="cuda:0")
    ck = torch.load(str(run_dir / "tensorboard" / ckpt_name), map_location="cpu", weights_only=False)
    runner.alg.actor.load_state_dict(ck["actor_state_dict"], strict=True); runner.alg.actor.eval()
    actor = runner.alg.actor; obj = env.scene["cube"]
    obs_td, _ = wrapped.reset(); obs = obs_td["actor"]
    cos_hist, tip_hist, palm_hist = [], [], []
    with torch.no_grad():
        for t in range(T):
            a = actor.mlp(obs) if hasattr(actor, "mlp") else actor(obs)
            obs_td, *_ = wrapped.step(a); obs = obs_td["actor"]
            q = obj.data.root_link_pose_w[:, 3:7]; qw, qx, qy, qz = q.unbind(-1)
            cos_hist.append((1 - 2 * (qx * qx + qy * qy)).clamp(-1, 1))
            if t >= 50:  # settle, then measure steady-state grip
                tf = read_force(env.unwrapped, "fingertip_cube_contact")
                pf = read_force(env.unwrapped, "palm_cube_contact")
                if tf is not None: tip_hist.append(tf.mean(-1) if tf.dim() == 2 else tf)
                if pf is not None: palm_hist.append(pf.mean(-1) if pf.dim() == 2 else pf)
    C = torch.stack(cos_hist, 0)
    held_cos = C[-50:].mean().item()
    tip = torch.stack(tip_hist, 0) if tip_hist else None
    palm = torch.stack(palm_hist, 0) if palm_hist else None
    res = dict(held_cos=held_cos,
               tip_mean=float(tip.mean()) if tip is not None else float("nan"),
               tip_peak=float(tip.max()) if tip is not None else float("nan"),
               palm_mean=float(palm.mean()) if palm is not None else float("nan"),
               palm_peak=float(palm.max()) if palm is not None else float("nan"))
    env.close()
    return res


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    specs = sys.argv[1:]
    if not specs:
        print(__doc__); return
    print(f"{'policy':10s} {'held_cos':>9s} {'tipF_mean':>10s} {'tipF_peak':>10s} {'palmF_mean':>11s} {'palmF_peak':>11s}")
    for spec in specs:
        name, rest = spec.split("=", 1)
        run, ck = rest.split(":")
        try:
            r = run_one(Path(run), ck)
            print(f"{name:10s} {r['held_cos']:9.3f} {r['tip_mean']:10.2f} {r['tip_peak']:10.2f} "
                  f"{r['palm_mean']:11.2f} {r['palm_peak']:11.2f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"{name:10s} ERROR {e}")


if __name__ == "__main__":
    main()
