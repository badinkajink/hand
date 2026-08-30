#!/usr/bin/env python3
"""Closed-loop observation ablation of a trained A->B pair.

`docs/rl/partial_observation_transfer.md` §4 asks for this BEFORE any distillation
training: roll the existing privileged policy out with individual observation blocks
disabled and measure what actually degrades. First-layer weight magnitude is not a
reliance test — correlated features and closed-loop compensation make it misleading —
so the intervention has to happen inside the rollout.

Three interventions, because "unavailable" has three honest meanings:

  zero     block -> 0. Removes the information but is itself out-of-distribution;
           a collapse here can be the OOD value, not the missing signal.
  freeze   block latched at its handoff-step value. This is the deploy proxy: what a
           controller sees if it takes one measurement and never updates it.
  shuffle  block permuted across the env batch. Keeps the marginal distribution exactly
           right and destroys only the correlation with THIS env's object.
  replay   block fed the values RECORDED from a baseline rollout at the same step. In
           distribution, causally decoupled, and deployable by construction: if a policy
           survives replay, that block was a clock and a clock can be shipped.

READ THE VARIANCE REPORT BEFORE THE ABLATION TABLE. shuffle and replay can only remove
information that across-env variance actually carries; with a deterministic spawn every
env sees nearly the same vector and permuting them is the identity map. `--variance-report`
prints, per block, the across-env spread against the across-time spread, and the summary
marks any block whose ablation could not have had teeth. Use `--spawn-jitter` to inject
real state variance when the nominal task has none.

The rollout is the continuous A->B handoff (`rl_demo_handoff_continuous.py`'s physics,
no reset at the seam), batched over `--num-envs` so every condition is a distribution —
GPU contact solves are non-deterministic and one rollout is not a measurement.

Example:
  uv run --extra rl --extra gpu python scripts/probe_obs_ablation.py \
      --policy-a results/rl/a10_.../tensorboard/model_270.pt \
      --policy-b results/rl/b33_.../tensorboard/model_270.pt \
      --morphology-run results/phase1/landscape/m05_ik_cem \
      --open-finger-from-keyframe --num-envs 32
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

# Blocks that a hardware controller genuinely cannot measure on the real_v1 bench:
# there is no object tracker, so anything derived from the object's pose is hidden.
# `ref_object_pose` is the frozen CEM reference (replayable open-loop, therefore
# KNOWN) and is listed separately for exactly that reason.
NAMED_GROUPS = {
    "hidden": ("object_pos", "object_pose_actual", "target_axis_misalign"),
    "objstate": ("object_pos", "object_pose_actual"),
    "deployable": ("joint_pos", "joint_vel", "ref_finger_qpos", "actions"),
}


def parse_condition(spec: str, term_names: list[str]) -> tuple[str, str, list[str]]:
    """'zero:hidden' / 'shuffle:object_pos+joint_vel' / 'none' -> (label, mode, terms)."""
    if spec == "none":
        return spec, "none", []
    mode, _, rhs = spec.partition(":")
    if mode not in ("zero", "freeze", "shuffle", "replay"):
        raise SystemExit(f"unknown intervention mode '{mode}' in '{spec}'")
    terms: list[str] = []
    for part in rhs.split("+"):
        terms.extend(NAMED_GROUPS.get(part, (part,)))
    unknown = [t for t in terms if t not in term_names]
    if unknown:
        raise SystemExit(f"'{spec}': no such observation term {unknown}; have {term_names}")
    return spec, mode, terms


def block_slices(env) -> dict[str, slice]:
    """Actor observation term -> its column slice, read off the live manager rather
    than hardcoded: the term order is dict-insertion order in `_build_observations`
    and silently shifts whenever a term is added."""
    om = env.observation_manager
    names = om.active_terms["actor"]
    dims = om.group_obs_term_dim["actor"]
    out, off = {}, 0
    for name, dim in zip(names, dims):
        width = int(np.prod(dim)) if isinstance(dim, (tuple, list)) else int(dim)
        out[name] = slice(off, off + width)
        off += width
    return out


def read_batched(env, sensor_name):
    """(force_mag, found) as (B, n_fingers) for every env, ordered [thumb, index, mid].
    `deploy.read_per_finger` is env-0 only and would collapse the distribution."""
    try:
        s = env.scene.sensors[sensor_name]
        f = getattr(s.data, "force", None)
        if f is None:
            return None, None
        mag = f.norm(dim=-1)
        if mag.dim() == 3:
            mag = mag.amax(dim=-1)
        found = getattr(s.data, "found", None)
        if found is not None and found.dim() == 3:
            found = found.amax(dim=-1)
        return mag, (found.float() if found is not None else torch.zeros_like(mag))
    except Exception:
        return None, None


def rollout(env, wrapped, actor_a, actor_b, a_obs_dim, *, handoff_step, total_steps,
            slices, mode, terms, ablate_from, replay_trace=None, record_obs=False):
    """One batched A->B rollout under one intervention. Returns per-env metrics,
    and (when `record_obs`) the full (T,B,obs_dim) actor-observation trace that the
    `replay` intervention and the variance report both feed on."""
    unwrapped = getattr(env, "unwrapped", env)
    obs_td, _ = wrapped.reset()
    n = obs_td["actor"].shape[0]
    dev = obs_td["actor"].device
    latched: dict[str, torch.Tensor] = {}

    z_hist, cos_hist, found_hist, force_hist, obs_hist = [], [], [], [], []
    with torch.no_grad():
        for step in range(total_steps):
            obs = obs_td["actor"]
            if record_obs:
                obs_hist.append(obs.clone())
            # --- the intervention, applied to the RAW observation the actor consumes.
            # Normalisation lives inside the actor, so mutating here is the faithful
            # "this number never reached the controller".
            if mode != "none" and step >= ablate_from:
                for t in terms:
                    sl = slices[t]
                    if mode == "zero":
                        obs[:, sl] = 0.0
                    elif mode == "freeze":
                        if t not in latched:
                            latched[t] = obs[:, sl].clone()
                        obs[:, sl] = latched[t]
                    elif mode == "shuffle":
                        obs[:, sl] = obs[torch.randperm(n, device=dev), sl]
                    elif mode == "replay":
                        # env i is fed the baseline value recorded for env (i+1) at this
                        # same step: a real observation of a real rollout, just not of
                        # the rollout it is now steering.
                        src = replay_trace[min(step, replay_trace.shape[0] - 1)]
                        obs[:, sl] = src.roll(1, dims=0)[:, sl].to(dev)

            if step < handoff_step:
                actions = (actor_a.act_inference(obs[:, :a_obs_dim])
                           if hasattr(actor_a, "act_inference") else actor_a.mlp(obs[:, :a_obs_dim]))
            else:
                actions = actor_b(obs_td)
            obs_td, *_ = wrapped.step(actions)

            pose = unwrapped.scene["cube"].data.root_link_pose_w  # (B,7) xyz + wxyz
            qx, qy = pose[:, 4], pose[:, 5]
            z_hist.append(pose[:, 2].clone())
            cos_hist.append(1.0 - 2.0 * (qx * qx + qy * qy))
            mag, found = read_batched(unwrapped, "fingertip_cube_contact")
            if mag is not None:
                force_hist.append(mag.clone())
                found_hist.append(found.clone())

    z = torch.stack(z_hist).cpu().numpy()          # (T,B)
    cos = torch.stack(cos_hist).cpu().numpy()      # (T,B)
    post = slice(handoff_step, None)
    out = {
        "min_z_post": z[post].min(axis=0),
        "final_cos": cos[-50:].mean(axis=0),
        "peak_cos": cos[post].max(axis=0),
        "final_z": z[-1],
    }
    if force_hist:
        f = torch.stack(force_hist).cpu().numpy()  # (T,B,3)
        g = torch.stack(found_hist).cpu().numpy()
        out["touch_frac"] = g[post].mean(axis=0)   # (B,3)
        out["force_mean"] = f[post].mean(axis=0)   # (B,3)
    if record_obs:
        out["_obs_trace"] = torch.stack(obs_hist)  # (T,B,D)
    return out


def variance_report(obs_trace, slices, handoff_step):
    """Per-block across-env spread vs across-time spread, over the post-handoff window.

    This is the ablation's own control. `shuffle` permutes the batch and `replay`
    swaps env identities; both can only destroy information that varies ACROSS ENVS.
    If a block's across-env sd is ~0 the permutation is the identity map and its row in
    the ablation table means nothing — the block was a clock, not a measurement.
    """
    t = obs_trace[handoff_step:].double()                     # (T,B,D)
    rows = {}
    for name, sl in slices.items():
        b = t[:, :, sl]
        across_env = b.std(dim=1).mean().item()               # spread between envs, mean over t
        across_time = b.mean(dim=1).std(dim=0).mean().item()  # spread over time of the env-mean
        rows[name] = {
            "across_env_sd": across_env,
            "across_time_sd": across_time,
            # <~5% means permuting envs barely changes the vector: no teeth.
            "ratio": across_env / across_time if across_time > 1e-9 else float("inf"),
            # A block with no across-env spread at all is identical in every env, so
            # every intervention that swaps envs is literally the identity map. The
            # two reference blocks are like this by construction — they are functions
            # of the step index alone.
            "constant_across_envs": across_env < 1e-9,
        }
    return rows


def summarize(label, m, hold_z):
    held = m["min_z_post"] > hold_z
    fc, pc = m["final_cos"], m["peak_cos"]
    row = {
        "condition": label,
        "hold_rate": float(held.mean()),
        "min_z_post_mean": float(m["min_z_post"].mean()),
        "final_cos_mean": float(fc.mean()),
        "final_cos_sd": float(fc.std()),
        "peak_cos_mean": float(pc.mean()),
        # Reorientation only counts on an object still in the hand — a dropped shaft
        # can read a fine peak_cos on its way to the floor (REORIENT_PRIMITIVE gotcha).
        "final_cos_held_mean": float(fc[held].mean()) if held.any() else float("nan"),
        "n_held": int(held.sum()),
        "n": int(held.size),
    }
    if "touch_frac" in m:
        row["touch_frac"] = [float(v) for v in m["touch_frac"].mean(axis=0)]
        row["force_mean"] = [float(v) for v in m["force_mean"].mean(axis=0)]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-a", type=Path, required=True)
    ap.add_argument("--policy-b", type=Path, required=True)
    ap.add_argument("--morphology-run", type=Path, required=True)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument("--handoff-step", type=int, default=40)
    ap.add_argument("--total-steps", type=int, default=240)
    ap.add_argument("--lift-delta", type=float, default=0.10)
    ap.add_argument("--finger-residual-scale", type=float, default=0.5)
    ap.add_argument("--finger-close-easing", type=str, default="ease_out_quad")
    ap.add_argument("--open-finger-from-keyframe", action="store_true")
    ap.add_argument("--no-contact-gate", action="store_true")
    ap.add_argument("--hold-ctrl-from-keyframe", type=str, default="")
    ap.add_argument("--hold-switch-from-sim-step", type=int, default=0)
    ap.add_argument("--hold-switch-steps", type=int, default=60)
    ap.add_argument("--hold-switch-min-z", type=float, default=0.0)
    ap.add_argument("--hold-z", type=float, default=0.05, help="min-z bar for 'held'")
    ap.add_argument("--ablate-from", type=int, default=None,
                    help="first step the intervention applies (default: the handoff step, "
                         "so Policy A always sees the truth and only B is blinded)")
    ap.add_argument("--actor-blind-terms", type=str, nargs="*", default=[],
                    help="Evaluate a policy that was TRAINED with these actor terms blinded. "
                         "Must match the run's config.yaml: a blind-trained actor read out in "
                         "a sighted env is fed live values where it only ever saw zeros, which "
                         "is the gotcha-#13 train/deploy parity bug in an observation coordinate.")
    ap.add_argument("--spawn-jitter", type=float, default=0.0,
                    help="object spawn xy jitter (m). The nominal task is deterministic, so "
                         "without this the envs barely differ and shuffle/replay have no teeth "
                         "— see the variance report. Note any value here is OOD for a policy "
                         "trained at 0.")
    ap.add_argument("--spawn-yaw-jitter", type=float, default=0.0,
                    help="object spawn yaw jitter (rad); same rationale as --spawn-jitter.")
    ap.add_argument("--conditions", type=str, nargs="+", default=[
        "none",
        "replay:hidden", "shuffle:hidden", "zero:hidden", "freeze:hidden",
        "replay:objstate", "replay:object_pose_actual", "replay:object_pos",
        "replay:target_axis_misalign", "replay:ref_object_pose",
        "replay:joint_vel", "replay:deployable",
    ])
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    from morphohand.rl.deploy import build_actor, ckpt_obs_dim, make_env_cfg
    from morphohand.tools.video_paths import tmp_dir

    with (args.morphology_run / "summary.json").open() as f:
        keyframe = json.load(f).get("keyframe", "open_short_manual")
    bfc = tuple(float(v) for v in
                np.load(args.morphology_run / "best_rollout.npz")["best_finger_ctrl"].reshape(-1))
    frozen = args.morphology_run / "frozen_scene.xml"
    work = tmp_dir("obs_ablation")
    work.mkdir(parents=True, exist_ok=True)
    ablate_from = args.handoff_step if args.ablate_from is None else args.ablate_from

    env_kwargs = dict(
        num_steps=args.total_steps, finger_residual_scale=args.finger_residual_scale,
        finger_close_easing=args.finger_close_easing,
        contact_gate_stability_rewards=not args.no_contact_gate, lift_delta=args.lift_delta,
        open_finger_from_keyframe=args.open_finger_from_keyframe, num_envs=args.num_envs,
        hold_ctrl_from_keyframe=args.hold_ctrl_from_keyframe,
        hold_switch_from_sim_step=args.hold_switch_from_sim_step,
        hold_switch_steps=args.hold_switch_steps,
        hold_switch_min_z=args.hold_switch_min_z)

    # Policy A in a throwaway env sized to ITS checkpoint (65 = native lift,
    # 66 = a B->A co-refined A), then the real 66-dim env for B.
    a_obs_dim = ckpt_obs_dim(args.policy_a)
    print(f"[abl] loading Policy A ({a_obs_dim}-dim)...")
    env_a, _, actor_a = build_actor(
        make_env_cfg(frozen, keyframe, args.morphology_run, bfc,
                     enable_target_axis=(a_obs_dim == 66), **{**env_kwargs, "num_steps": 10}),
        args.policy_a, work)
    env_a.close()

    print(f"[abl] building 66-dim handoff env, {args.num_envs} envs...")
    cfg_b = make_env_cfg(frozen, keyframe, args.morphology_run, bfc,
                         enable_target_axis=True, **env_kwargs)
    import dataclasses as _dc
    if args.spawn_jitter or args.spawn_yaw_jitter:
        cfg_b = _dc.replace(cfg_b, cube_spawn_xy_jitter=args.spawn_jitter,
                            cube_spawn_yaw_jitter=args.spawn_yaw_jitter)
        print(f"[abl] spawn jitter xy={args.spawn_jitter} yaw={args.spawn_yaw_jitter} "
              f"(OOD for a policy trained at 0 — read the baseline row first)")
    if args.actor_blind_terms:
        cfg_b = _dc.replace(cfg_b, actor_blind_terms=tuple(args.actor_blind_terms))
        print(f"[abl] actor blinded to {list(args.actor_blind_terms)} (matching its training env); "
              f"ablation conditions naming those terms are no-ops by construction")
    env, wrapped, actor_b = build_actor(cfg_b, args.policy_b, work)
    unwrapped = getattr(env, "unwrapped", env)
    slices = block_slices(unwrapped)
    print("[abl] actor observation layout: "
          + "  ".join(f"{k}[{v.start}:{v.stop}]" for k, v in slices.items()))

    conditions = [parse_condition(c, list(slices)) for c in args.conditions]
    if any(m == "replay" for _, m, _ in conditions) and \
            not any(m == "none" for _, m, _ in conditions):
        raise SystemExit("replay conditions need a 'none' baseline in --conditions to record from")

    rows, replay_trace, var_rows = [], None, None
    for label, mode, terms in conditions:
        m = rollout(env, wrapped, actor_a, actor_b, a_obs_dim,
                    handoff_step=args.handoff_step, total_steps=args.total_steps,
                    slices=slices, mode=mode, terms=terms, ablate_from=ablate_from,
                    replay_trace=replay_trace, record_obs=(mode == "none"))
        if mode == "none":
            trace = m.pop("_obs_trace")
            replay_trace = trace                      # (T,B,D), the source for `replay`
            var_rows = variance_report(trace, slices, args.handoff_step)
            print("\n[abl] ---- VARIANCE REPORT (post-handoff) — does the intervention have teeth? ----")
            print(f"[abl] {'block':22s} {'across-env sd':>14s} {'across-time sd':>15s} {'ratio':>8s}")
            for k, v in var_rows.items():
                flag = ("  <- NO TEETH: identical in every env (a clock)"
                        if v["constant_across_envs"] else
                        "  <- weak: across-env spread under 5% of across-time"
                        if v["ratio"] < 0.05 else "")
                print(f"[abl] {k:22s} {v['across_env_sd']:14.5f} "
                      f"{v['across_time_sd']:15.5f} {v['ratio']:8.3f}{flag}")
            print()
        row = summarize(label, m, args.hold_z)
        # A row is uninformative only if EVERY block it touched was untouchable.
        row["no_teeth"] = bool(var_rows and terms and mode in ("shuffle", "replay") and
                               all(var_rows[t]["constant_across_envs"] or
                                   var_rows[t]["ratio"] < 0.05 for t in terms))
        rows.append(row)
        print(f"[abl] {label:34s} hold {row['hold_rate']:.2f}  "
              f"min_z {row['min_z_post_mean']:.3f}  "
              f"final_cos {row['final_cos_mean']:+.3f}±{row['final_cos_sd']:.3f}  "
              f"peak_cos {row['peak_cos_mean']:+.3f}  "
              f"cos|held {row['final_cos_held_mean']:+.3f}")
    env.close()

    base = next((r for r in rows if r["condition"] == "none"), None)
    print("\n" + "=" * 96)
    print(f"{'condition':34s} {'hold':>5s} {'min_z':>6s} {'finalcos':>9s} {'Δcos':>7s} {'peak':>6s}")
    print("-" * 96)
    for r in rows:
        d = (r["final_cos_mean"] - base["final_cos_mean"]) if base else float("nan")
        note = "  (no teeth)" if r.get("no_teeth") else ""
        print(f"{r['condition']:34s} {r['hold_rate']:5.2f} {r['min_z_post_mean']:6.3f} "
              f"{r['final_cos_mean']:+9.3f} {d:+7.3f} {r['peak_cos_mean']:+6.3f}{note}")
    print("=" * 96)
    if any(r.get("no_teeth") for r in rows):
        print("(no teeth) = every ablated block had across-env sd < 5% of its across-time sd,")
        print("so permuting/replacing across envs barely changed the vector. Those rows are")
        print("evidence the block is a CLOCK, not evidence the policy ignores it.")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "policy_a": str(args.policy_a), "policy_b": str(args.policy_b),
            "morphology_run": str(args.morphology_run), "num_envs": args.num_envs,
            "handoff_step": args.handoff_step, "ablate_from": ablate_from,
            "spawn_jitter": args.spawn_jitter, "spawn_yaw_jitter": args.spawn_yaw_jitter,
            "obs_layout": {k: [v.start, v.stop] for k, v in slices.items()},
            "variance": var_rows, "rows": rows,
        }, indent=1))
        print(f"[abl] -> {args.output}")


if __name__ == "__main__":
    main()
