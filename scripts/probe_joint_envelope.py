"""What joint envelope does a trained policy actually USE, and does the hardware have it?

Every perp result to date was produced against MJCF joint ranges that were authored for
simulation convenience, not copied from the prototype. They are wider than the hardware in
every DOF (sim yaw +/-63 deg vs the servo's +/-45; sim MCP up to 143-180 deg vs 0-110; sim PIP
-103..+29 vs 0-100). A policy is only a sim2real candidate if the trajectory it commands fits
inside the envelope the hardware can actually reach -- and "the joint range in the XML is wider
than the servo" says nothing on its own, because the policy may never go near the limit.

So this asks the rollout instead of the XML. It reports, per actuated finger joint, the range
the policy SWEEPS over N rollouts and the peak rate it demands, against a hardware table, and
prints a per-joint verdict. Rate matters as much as range: a servo has a slew limit, and a
policy that steps 300 deg/s is not deployable no matter how modest its travel.

Read the verdict as necessary, never sufficient -- it is a kinematic envelope check, not a
torque, bandwidth or contact-fidelity check.

Usage:
  uv run --extra rl --extra gpu python scripts/probe_joint_envelope.py \
      --policy results/rl/<run>/tensorboard/model_338.pt \
      --morphology-run results/phase1/perp/perp_v1 \
      --closed-ctrl-from-keyframe closed --open-finger-from-keyframe \
      --lift-delta 0.14 --steps 600 --n 32
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from morphohand.rl.deploy import (
    act, act_b, build_actor, ckpt_obs_dim, finger_ctrl_from_keyframe, make_env_cfg,
)
from morphohand.tools.video_paths import tmp_dir

# hand_paper/main.tex Table I ("Degrees of Freedom per Finger"). Degrees.
# Yaw is the ABD/ADD servo; the perp topology's 90 deg opposition is NOT this joint -- it is a
# fixed mount rotation baked into the body quat, so it needs a physical remount, not travel here.
HARDWARE_DEG = {
    "yaw": (-45.0, 45.0),
    "mcp": (0.0, 110.0),
    "pip": (0.0, 100.0),
}
# No slew figure is published for the servos, so this is a placeholder to be replaced with the
# measured value; it is reported as a number to compare against, never as a pass/fail.
DEFAULT_MAX_RATE_DEG_S = 300.0


def hw_limits(joint_name: str):
    for key, lim in HARDWARE_DEG.items():
        if joint_name.endswith(key):
            return lim
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--morphology-run", type=Path, required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lift-delta", type=float, default=0.14)
    ap.add_argument("--finger-residual-scale", type=float, default=0.5)
    ap.add_argument("--open-finger-from-keyframe", action="store_true")
    ap.add_argument("--closed-ctrl-from-keyframe", default=None,
                    help="MUST match how the policy was trained (gotcha #13)")
    ap.add_argument("--max-rate-deg-s", type=float, default=DEFAULT_MAX_RATE_DEG_S)
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    run = args.morphology_run.resolve()
    summ = json.loads((run / "summary.json").read_text())
    frozen = run / "frozen_scene.xml"
    if args.closed_ctrl_from_keyframe:
        bfc = finger_ctrl_from_keyframe(frozen, args.closed_ctrl_from_keyframe)
    else:
        bfc = tuple(float(v)
                    for v in np.load(run / "best_rollout.npz")["best_finger_ctrl"].reshape(-1))

    obs_dim = ckpt_obs_dim(args.policy)
    is_b = obs_dim == 66
    cfg = make_env_cfg(frozen, summ["keyframe"], run, bfc, enable_target_axis=is_b,
                       num_steps=args.steps,
                       finger_residual_scale=args.finger_residual_scale,
                       lift_delta=args.lift_delta,
                       open_finger_from_keyframe=args.open_finger_from_keyframe,
                       num_envs=args.n)
    env, wrapped, actor = build_actor(cfg, args.policy, tmp_dir("jointenv"))
    obs_td, _ = wrapped.reset()

    robot = env.unwrapped.scene["robot"]
    names = list(robot.joint_names)
    finger_ids = [i for i, n in enumerate(names) if hw_limits(n) is not None]
    finger_names = [names[i] for i in finger_ids]
    dt = float(env.unwrapped.step_dt)

    qpos = np.zeros((args.steps, args.n, len(finger_ids)), dtype=np.float32)
    with torch.no_grad():
        for s in range(args.steps):
            obs = obs_td["actor"]
            actions = act_b(actor, obs_td, False) if is_b else act(actor, obs[:, :obs_dim])
            obs_td, *_ = wrapped.step(actions)
            qpos[s] = robot.data.joint_pos[:, finger_ids].cpu().numpy()
    env.close()

    deg = np.degrees(qpos)
    # Per-step rate. Envs are stepped together and never reset mid-rollout here, so a diff
    # across the step axis is a true trajectory derivative and not a reset discontinuity.
    rate = np.abs(np.diff(deg, axis=0)) / dt

    rows, out = [], {"label": args.label or args.policy.parent.parent.name,
                     "policy": str(args.policy), "n": args.n, "steps": args.steps,
                     "dt": dt, "joints": {}}
    any_range_violation = False
    offset_joints = []
    for j, name in enumerate(finger_names):
        lo, hi = float(deg[:, :, j].min()), float(deg[:, :, j].max())
        p99 = float(np.percentile(rate[:, :, j], 99.0))
        rmax = float(rate[:, :, j].max())
        hlo, hhi = hw_limits(name)
        over_lo, over_hi = max(0.0, hlo - lo), max(0.0, hi - hhi)
        bad = over_lo > 0.0 or over_hi > 0.0
        # A band that sits outside the hardware interval but is NARROWER than it is a zero/sign
        # convention difference between the MJCF and the hardware table, not travel the servo
        # lacks -- the thumb is mounted mirrored, so its MJCF flexion runs negative. Called out
        # separately because it is fixed by a mount offset, and reporting it as a reach failure
        # would send someone looking for a bigger servo.
        span_fits = (hi - lo) <= (hhi - hlo)
        if not bad:
            verdict = "OK"
        elif span_fits:
            verdict = f"OFFSET {max(over_lo, over_hi):.0f} deg (travel fits)"
            offset_joints.append(name)
        else:
            verdict = f"OVER by {max(over_lo, over_hi):.1f} deg"
            any_range_violation = True
        rows.append((name, lo, hi, hlo, hhi, p99, rmax, verdict))
        out["joints"][name] = {
            "used_deg": [lo, hi], "travel_deg": hi - lo, "hardware_deg": [hlo, hhi],
            "over_low_deg": over_lo, "over_high_deg": over_hi,
            "travel_fits": span_fits,
            "rate_p99_deg_s": p99, "rate_max_deg_s": rmax, "verdict": verdict,
        }

    w = max(len(n) for n in finger_names)
    print(f"\n┌── joint envelope: {out['label']}   N={args.n} x {args.steps} steps, dt={dt:.4f}s")
    print(f"│  {'joint':{w}s}  {'policy uses (deg)':>20s}  {'hardware (deg)':>16s}  "
          f"{'rate p99':>9s}  {'rate max':>9s}   verdict")
    for name, lo, hi, hlo, hhi, p99, rmax, verdict in rows:
        print(f"│  {name:{w}s}  [{lo:+7.1f},{hi:+7.1f}]  [{hlo:+6.1f},{hhi:+6.1f}]  "
              f"{p99:8.1f}  {rmax:8.1f}   {verdict}")
    fast = [n for n, *_, p99, _, _ in rows if p99 > args.max_rate_deg_s]
    print(f"│  travel: {'FAIL — needs more range than the prototype has' if any_range_violation else 'PASS — every joint travels less than the prototype allows'}")
    if offset_joints:
        print(f"│  offset: {len(offset_joints)} joint(s) sit outside the hardware interval with the "
              f"travel fitting — a mount/zero convention, not a servo: {', '.join(offset_joints)}")
    print(f"│  rate:   p99 over {args.max_rate_deg_s:.0f} deg/s on {len(fast)} joint(s)"
          f"{': ' + ', '.join(fast) if fast else ''}")
    print("└" + "─" * 60)
    out["range_ok"] = not any_range_violation
    out["offset_joints"] = offset_joints
    out["fast_joints"] = fast
    out["max_rate_deg_s_ref"] = args.max_rate_deg_s

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"[envelope] -> {args.json_out}")


if __name__ == "__main__":
    main()
