"""N-rollout evaluation for a reorient policy: success + stability metrics as a DISTRIBUTION.

Why this exists. `policy_healthcheck.py` rolls ONE episode and grades it. On the perp
topology that turned out not to be reproducible: three rollouts of the same deterministic
checkpoint, with stochastic sampling off and spawn jitter at zero, ended three different
ways (held at step 400 / lost at 430 / lost before 400). The actions are deterministic, so
the spread comes from the simulator — parallel contact solves do not reduce in a fixed
order on GPU. A single rollout therefore cannot tell you whether a policy holds; it tells
you what happened once.

So every number here is over N envs stepped together, and the headline is a RATE plus a
spread, not a value. Cheap: N rollouts cost about the same wall-clock as one, because the
envs are batched.

Metrics, all of them about the task rather than the reward:
  align_rate     fraction of envs that ever reach cos >= --align-thresh
  hold_rate      fraction still HOLDING at the end (the number that matters -- a shaft
                 standing on the floor reads cos +1.000 and is not a hold)
  t_align        steps to first reach the threshold (speed of the reorient)
  hold_steps     consecutive steps aligned AND held, per env (stability of the hold)
  final_cos/z    where it ended up
  drop_step      when the object was lost, for the envs that lost it

"Held" is asked of the physics, not inferred from height: total fingertip force > --held-min-n
AND object above --floor-z. That distinction is load-bearing on this topology, where the
documented failure is the shaft reaching vertical by sliding OUT of the pinch and ending
upright on the floor -- perfect cos, zero grip.

Usage:
  uv run --extra rl --extra gpu python scripts/policy_eval_suite.py \
      --policy results/rl/<run>/tensorboard/model_338.pt \
      --morphology-run results/phase1/perp/perp_v1 \
      --closed-ctrl-from-keyframe closed --open-finger-from-keyframe \
      --lift-delta 0.14 --steps 500 --n 64 --plot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from morphohand.rl.deploy import (
    act, act_b, build_actor, ckpt_obs_dim, finger_ctrl_from_keyframe, make_env_cfg,
)
from morphohand.tools.video_paths import tmp_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--morphology-run", type=Path, required=True)
    ap.add_argument("--n", type=int, default=64, help="parallel rollouts")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--align-thresh", type=float, default=0.9,
                    help="cos counted as 'reoriented'")
    ap.add_argument("--held-min-n", type=float, default=0.5,
                    help="total fingertip force (N) below which the object is NOT held")
    ap.add_argument("--floor-z", type=float, default=0.06,
                    help="object z below which it is resting on the floor, not held")
    ap.add_argument("--lift-delta", type=float, default=0.14)
    ap.add_argument("--finger-residual-scale", type=float, default=0.5)
    ap.add_argument("--open-finger-from-keyframe", action="store_true")
    ap.add_argument("--closed-ctrl-from-keyframe", default=None,
                    help="MUST match how the policy was trained (gotcha #13)")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=None, help="write a summary PNG here")
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
    env, wrapped, actor = build_actor(cfg, args.policy, tmp_dir("evalsuite"))
    obs_td, _ = wrapped.reset()

    N = args.n
    cos_t = np.zeros((args.steps, N), dtype=np.float32)
    z_t = np.zeros((args.steps, N), dtype=np.float32)
    grip_t = np.zeros((args.steps, N), dtype=np.float32)

    sensor = "fingertip_cube_contact"
    with torch.no_grad():
        for s in range(args.steps):
            obs = obs_td["actor"]
            actions = act_b(actor, obs_td, False) if is_b else act(actor, obs[:, :obs_dim])
            obs_td, *_ = wrapped.step(actions)
            pose = env.unwrapped.scene["cube"].data.root_link_pose_w      # (N, 7)
            qx, qy = pose[:, 4], pose[:, 5]
            cos_t[s] = (1.0 - 2.0 * (qx * qx + qy * qy)).cpu().numpy()
            z_t[s] = pose[:, 2].cpu().numpy()
            f = env.unwrapped.scene.sensors[sensor].data.force
            grip_t[s] = (f.norm(dim=-1).sum(dim=-1).cpu().numpy()
                         if f is not None else np.zeros(N))
    env.close()

    held_t = (grip_t > args.held_min_n) & (z_t > args.floor_z)      # (T, N)
    aligned_t = cos_t >= args.align_thresh
    good_t = aligned_t & held_t

    ever_aligned = aligned_t.any(axis=0)
    held_final = held_t[-1]
    # first step each env crosses the threshold (nan where it never does)
    t_align = np.where(ever_aligned, aligned_t.argmax(axis=0), np.nan)
    # longest run of aligned-AND-held per env
    hold_steps = np.zeros(N, dtype=int)
    for e in range(N):
        best = cur = 0
        for v in good_t[:, e]:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        hold_steps[e] = best
    lost = held_t.any(axis=0) & ~held_final
    drop_step = np.full(N, np.nan)
    for e in np.nonzero(lost)[0]:
        h = held_t[:, e]
        drop_step[e] = int(np.nonzero(h)[0].max()) + 1

    def stat(a):
        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return "—"
        return f"{a.mean():.3f} ± {a.std():.3f}  [{a.min():.3f}, {a.max():.3f}]"

    label = args.label or args.policy.parent.parent.name
    print(f"\n┌── eval suite: {label}   N={N} rollouts x {args.steps} steps")
    print(f"│  align_rate  (ever cos>={args.align_thresh})   "
          f"{ever_aligned.mean():6.1%}  ({int(ever_aligned.sum())}/{N})")
    print(f"│  HOLD_RATE   (still held at end)      "
          f"{held_final.mean():6.1%}  ({int(held_final.sum())}/{N})")
    print(f"│  success     (aligned AND held)       "
          f"{(ever_aligned & held_final).mean():6.1%}")
    print(f"│  t_align     (steps to threshold)     {stat(t_align)}")
    print(f"│  hold_steps  (aligned+held run)       {stat(hold_steps)}")
    print(f"│  final_cos                            {stat(cos_t[-1])}")
    print(f"│  final_z                              {stat(z_t[-1])}")
    print(f"│  peak_cos                             {stat(cos_t.max(axis=0))}")
    if lost.any():
        print(f"│  drop_step   ({int(lost.sum())} envs lost it)      {stat(drop_step)}")
    print("└" + "─" * 60)

    out = dict(
        label=label, n=N, steps=args.steps, align_thresh=args.align_thresh,
        align_rate=float(ever_aligned.mean()), hold_rate=float(held_final.mean()),
        success_rate=float((ever_aligned & held_final).mean()),
        t_align_mean=float(np.nanmean(t_align)) if ever_aligned.any() else None,
        hold_steps_mean=float(hold_steps.mean()), hold_steps_sd=float(hold_steps.std()),
        final_cos_mean=float(cos_t[-1].mean()), final_cos_sd=float(cos_t[-1].std()),
        final_z_mean=float(z_t[-1].mean()),
        peak_cos_mean=float(cos_t.max(axis=0).mean()),
        drop_step_mean=float(np.nanmean(drop_step)) if lost.any() else None,
        n_lost=int(lost.sum()),
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2))
        print(f"[eval] -> {args.json_out}")

    if args.plot:
        _plot(args.plot, label, cos_t, z_t, grip_t, held_t, args)
        print(f"[eval] -> {args.plot}")


def _plot(path: Path, label, cos_t, z_t, grip_t, held_t, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, N = cos_t.shape
    t = np.arange(T)
    fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    def band(a, axis, color, ylabel, hline=None):
        med = np.median(a, axis=1)
        lo, hi = np.percentile(a, 10, axis=1), np.percentile(a, 90, axis=1)
        axis.fill_between(t, lo, hi, color=color, alpha=0.25, label="10-90%")
        axis.plot(t, med, color=color, lw=2, label="median")
        if hline is not None:
            axis.axhline(hline, ls="--", lw=1, color="k", alpha=0.6)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)

    band(cos_t, ax[0], "tab:blue", "alignment cos", args.align_thresh)
    ax[0].set_title(f"{label}   N={N} rollouts")
    ax[0].legend(loc="lower right", fontsize=8)
    band(z_t, ax[1], "tab:green", "object z (m)", args.floor_z)
    band(grip_t, ax[2], "tab:orange", "total grip (N)", args.held_min_n)
    ax[3].plot(t, held_t.mean(axis=1), color="tab:red", lw=2)
    ax[3].set_ylabel("fraction HELD")
    ax[3].set_xlabel("policy step")
    ax[3].set_ylim(-0.05, 1.05)
    ax[3].grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
