#!/usr/bin/env python3
"""Render one deterministic rollout of a reorient checkpoint at a chosen resolution.

    uv run --extra rl --extra gpu python scripts/rl_render_rollout.py \
        --policy results/rl/<run>/tensorboard/model_270.pt \
        --morphology-run results/phase1/real_v1/20260916-sv1_u0308_b050_cal \
        --closed-ctrl-from-keyframe open_ik --open-finger-from-keyframe --lift-delta 0.1 \
        --steps 250 --out docs/experiments/<folder>/<name>.mp4

Same env construction as `policy_eval_suite.py` (so the rollout is the one the numbers
describe), one env, the in-training viewer replaced by a close camera. Writes the mp4 and a
CSV of the tool's signed cos, height and per-pad force per policy step.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from morphohand.rl.deploy import (act, act_b, build_actor, ckpt_obs_dim, finger_ctrl_from_keyframe,
                                  make_env_cfg, run_env_overrides)
from morphohand.tools.video_paths import tmp_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--morphology-run", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lift-delta", type=float, default=0.10)
    ap.add_argument("--finger-residual-scale", type=float, default=0.5)
    ap.add_argument("--open-finger-from-keyframe", action="store_true")
    ap.add_argument("--closed-ctrl-from-keyframe", default=None)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cam", default="0.32,120,-12", help="distance m, azimuth deg, elevation deg")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--residual-from-step", type=int, default=None,
                    help="finger_residual_active_from_step; default = the run's config.yaml")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the mean")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import json
    run = args.morphology_run.resolve()
    summ = json.loads((run / "summary.json").read_text())
    frozen = run / "frozen_scene.xml"
    if args.closed_ctrl_from_keyframe:
        bfc = finger_ctrl_from_keyframe(frozen, args.closed_ctrl_from_keyframe)
    else:
        bfc = tuple(float(v) for v in np.load(run / "best_rollout.npz")["best_finger_ctrl"].reshape(-1))
    obs_dim = ckpt_obs_dim(args.policy)
    is_b = obs_dim == 66
    trained = run_env_overrides(args.policy)
    residual_from = args.residual_from_step if args.residual_from_step is not None \
        else int(trained.get("finger_residual_active_from_step", 0))
    print(f"[render] residual active from step {residual_from}, reorient reward from step "
          f"{trained.get('reorient_start_step', 10)} ({'run config.yaml' if trained else 'defaults'})")
    cfg = make_env_cfg(frozen, summ["keyframe"], run, bfc, enable_target_axis=is_b, num_steps=args.steps,
                       finger_residual_scale=args.finger_residual_scale, lift_delta=args.lift_delta,
                       open_finger_from_keyframe=args.open_finger_from_keyframe, num_envs=1,
                       finger_residual_active_from_step=residual_from,
                       reorient_start_step=int(trained.get("reorient_start_step", 10)),
                       lift_phase_start_step=trained.get("lift_phase_start_step"))
    dist, az, el = (float(v) for v in args.cam.split(","))
    cfg.viewer_width, cfg.viewer_height = args.width, args.height
    cfg.viewer_distance, cfg.viewer_azimuth, cfg.viewer_elevation = dist, az, el
    env, wrapped, actor = build_actor(cfg, args.policy, tmp_dir("render"), render_mode="rgb_array")
    obs_td, _ = wrapped.reset()
    frames, rows = [], []
    sensor = "fingertip_cube_contact"
    with torch.no_grad():
        for s in range(args.steps):
            obs = obs_td["actor"]
            actions = act_b(actor, obs_td, args.stochastic) if is_b else act(actor, obs[:, :obs_dim])
            obs_td, *_ = wrapped.step(actions)
            pose = env.unwrapped.scene["cube"].data.root_link_pose_w
            qx, qy = float(pose[0, 4]), float(pose[0, 5])
            f = env.unwrapped.scene.sensors[sensor].data.force
            mag = f.norm(dim=-1)[0].cpu().numpy() if f is not None else np.zeros(3)
            rows.append([s, round(1.0 - 2.0 * (qx * qx + qy * qy), 4), round(float(pose[0, 2]), 4)]
                        + [round(float(v), 3) for v in mag[:3]])
            frame = env.unwrapped.render()
            if frame is not None:
                frames.append(np.asarray(frame).copy())
    env.close()
    import imageio
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(args.out), frames, fps=args.fps, codec="libx264", quality=7)
    with open(args.out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "cos", "z", "f_thumb", "f_index", "f_middle"])
        w.writerows(rows)
    print(f"wrote {args.out} ({len(frames)} frames) and {args.out.with_suffix('.csv')}; "
          f"final cos {rows[-1][1]:+.3f} z {rows[-1][2]:.3f} pads {rows[-1][3:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
