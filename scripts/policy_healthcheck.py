"""Single-policy trajectory-health check — flag a degenerate grasp/reorient policy.

Rolls ONE policy (Policy A lift, or a Policy B reorienter) through a full deterministic
episode, logs the per-step trajectory, and prints the PASS/WARN/FAIL scorecard from
`morphohand.rl.trajectory_health` (late finger / drop / jitter / idle-finger /
de-centering / over-clamp). This is the reusable "is this policy actually good, or does it
just look good on aggregate reward" gate — the check that would have caught the m05
delayed-finger 2-finger grasp automatically.

For the A->B handoff, rl_demo_handoff_continuous.py already prints the same scorecard.

Run:
  uv run --extra rl --extra gpu python scripts/policy_healthcheck.py \
    --policy results/rl/<run>/tensorboard/model_N.pt \
    --morphology-run results/phase1/landscape/m05_ik_cem \
    --open-finger-from-keyframe --lift-delta 0.05 [--grasp-end 48 --hold-start 63]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

from morphohand.rl.deploy import act, act_b, build_actor, ckpt_obs_dim, make_env_cfg, read_per_finger
from morphohand.rl.trajectory_health import characterize_trajectory, format_scorecard

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--morphology-run", type=Path, required=True)
    ap.add_argument("--total-steps", type=int, default=200)
    ap.add_argument("--grasp-end", type=int, default=48,
                    help="policy step the lift/grasp is complete (late-finger judged here)")
    ap.add_argument("--hold-start", type=int, default=63,
                    help="policy step the steady hold begins (balance/drift/jitter here)")
    ap.add_argument("--lift-delta", type=float, default=0.05)
    ap.add_argument("--open-finger-from-keyframe", action="store_true")
    ap.add_argument("--finger-residual-scale", type=float, default=0.5)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    run = args.morphology_run.resolve()
    summ = json.loads((run / "summary.json").read_text())
    frozen = run / "frozen_scene.xml"
    keyframe = summ["keyframe"]
    bfc = tuple(float(v) for v in np.load(run / "best_rollout.npz")["best_finger_ctrl"].reshape(-1))

    # detect obs dim: 65 = Policy A (lift), 66 = a reorienter (target_axis obs)
    obs_dim = ckpt_obs_dim(args.policy)
    is_b = obs_dim == 66
    # temp renders live under logs/ (gitignored), never in the tracked docs tree
    from morphohand.tools.video_paths import tmp_dir
    work = tmp_dir("healthcheck")

    cfg = make_env_cfg(frozen, keyframe, run, bfc, enable_target_axis=is_b,
                       num_steps=args.total_steps, finger_residual_scale=args.finger_residual_scale,
                       lift_delta=args.lift_delta,
                       open_finger_from_keyframe=args.open_finger_from_keyframe)
    env, wrapped, actor = build_actor(cfg, args.policy, work)
    obs_td, _ = wrapped.reset()
    obs = obs_td["actor"]
    full = {"found": [], "force": [], "z": [], "cos": [], "x": [], "y": [], "axis": []}
    with torch.no_grad():
        for _ in range(args.total_steps):
            actions = act_b(actor, obs_td, False) if is_b else act(actor, obs[:, :obs_dim])
            obs_td, *_ = wrapped.step(actions)
            obs = obs_td["actor"]
            pose = env.unwrapped.scene["cube"].data.root_link_pose_w[0]
            x, y, z = float(pose[0]), float(pose[1]), float(pose[2])
            qw, qx, qy, qz = (float(pose[3]), float(pose[4]), float(pose[5]), float(pose[6]))
            cos = 1.0 - 2.0 * (qx * qx + qy * qy)
            ff, fg = read_per_finger(env.unwrapped, "fingertip_cube_contact")
            full["found"].append(fg if fg is not None else [0., 0., 0.])
            full["force"].append(ff if ff is not None else [0., 0., 0.])
            full["z"].append(z); full["cos"].append(cos); full["x"].append(x); full["y"].append(y)
            full["axis"].append((2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), cos))
    env.close()

    axis = np.asarray(full["axis"])
    dots = np.clip((axis[1:] * axis[:-1]).sum(1), -1.0, 1.0)
    angvel = np.concatenate([[0.0], np.arccos(dots) / 0.02])
    sc = characterize_trajectory(
        finger_found=full["found"], finger_force=full["force"], obj_z=full["z"],
        obj_cos=full["cos"], obj_xy=np.stack([full["x"], full["y"]], axis=1),
        obj_angvel=angvel, grasp_end=args.grasp_end, hold_start=args.hold_start)
    title = args.title or f"{args.policy.parent.parent.name} ({'B' if is_b else 'A'})"
    print(format_scorecard(sc, title=title))
    out = args.json_out or args.policy.with_suffix(".health.json")
    out.write_text(json.dumps({**sc.as_dict(), "policy": str(args.policy), "title": title}, indent=1))
    print(f"[health] -> {out}")


if __name__ == "__main__":
    main()
