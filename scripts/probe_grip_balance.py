"""Probe: PER-FINGER contact force + contact-found fraction in a policy's own standalone
reorient env. Tests the hypothesis that the "excessive grip force" is actually a degenerate
2-point pinch: one of index/middle loses contact, the other barely touches, so the THUMB
carries everything (huge thumb force) instead of a balanced tripod.

The aggregate probe (probe_grip_force.py) averages over the 3 fingertips and HIDES this.
The fingertip_cube_contact sensor matches bodies (thumb_tip, index_tip, middle_tip) in that
order, reduce="none" -> per-finger force is directly available.

Usage:
  uv run python scripts/probe_grip_balance.py \
    b34_t20=results/rl/bx_20260613-0909-policyB_b33iter_forcefloor_t20:model_405.pt \
    B4=results/rl/b04_20260603-1746-policyB_p2_lateral_only:model_541.pt
"""
from __future__ import annotations
import sys, dataclasses, os
from pathlib import Path
import numpy as np, torch, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FINGERS = ["thumb", "index", "middle"]


def per_finger(env_unwrapped, sensor_name):
    """Return (force_mag, found) each shape (B, 3) in [thumb, index, middle] order, or (None, None)."""
    try:
        s = env_unwrapped.scene.sensors[sensor_name]
        f = getattr(s.data, "force", None)
        found = getattr(s.data, "found", None)
        if f is None:
            return None, None
        # f: (B, n_bodies, [slots,] 3). norm the xyz; collapse any slots dim by max.
        mag = f.norm(dim=-1)                       # (B, n_bodies[, slots])
        if mag.dim() == 3:
            mag = mag.amax(dim=-1)                 # (B, n_bodies)
        if found is not None and found.dim() == 3:
            found = found.amax(dim=-1)
        return mag, (found.float() if found is not None else None)
    except Exception:
        return None, None


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
    cos_hist, force_hist, found_hist = [], [], []
    with torch.no_grad():
        for t in range(T):
            a = actor.mlp(obs) if hasattr(actor, "mlp") else actor(obs)
            obs_td, *_ = wrapped.step(a); obs = obs_td["actor"]
            q = obj.data.root_link_pose_w[:, 3:7]; qw, qx, qy, qz = q.unbind(-1)
            cos_hist.append((1 - 2 * (qx * qx + qy * qy)).clamp(-1, 1))
            if t >= 50:  # settle, then measure steady-state
                mag, found = per_finger(env.unwrapped, "fingertip_cube_contact")
                if mag is not None: force_hist.append(mag)
                if found is not None: found_hist.append(found)
    held_cos = torch.stack(cos_hist, 0)[-50:].mean().item()
    F = torch.stack(force_hist, 0) if force_hist else None       # (T', B, 3)
    G = torch.stack(found_hist, 0) if found_hist else None
    env.close()
    if F is None:
        return dict(held_cos=held_cos, force=None)
    force_mean = F.reshape(-1, 3).mean(0).cpu().numpy()          # per-finger mean force
    found_frac = G.reshape(-1, 3).mean(0).cpu().numpy() if G is not None else np.full(3, np.nan)
    # "carried by thumb" share: thumb_force / total_force
    total = float(force_mean.sum()) or 1.0
    return dict(held_cos=held_cos, force_mean=force_mean, found_frac=found_frac,
                thumb_share=float(force_mean[0]) / total)


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    specs = sys.argv[1:]
    if not specs:
        print(__doc__); return
    print(f"{'policy':12s} {'held_cos':>8s} | {'thumbF':>7s} {'indexF':>7s} {'midF':>7s} "
          f"| {'thumb%':>6s} | contact-found: {'thumb':>6s} {'index':>6s} {'mid':>6s}")
    for spec in specs:
        name, rest = spec.split("=", 1)
        run, ck = rest.split(":")
        try:
            r = run_one(Path(run), ck)
            if r.get("force_mean") is None:
                print(f"{name:12s} {r['held_cos']:8.3f} | NO FORCE DATA"); continue
            fm, ff = r["force_mean"], r["found_frac"]
            print(f"{name:12s} {r['held_cos']:8.3f} | {fm[0]:7.2f} {fm[1]:7.2f} {fm[2]:7.2f} "
                  f"| {100*r['thumb_share']:5.0f}% | contact-found: {ff[0]:6.2f} {ff[1]:6.2f} {ff[2]:6.2f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"{name:12s} ERROR {e}")


if __name__ == "__main__":
    main()
