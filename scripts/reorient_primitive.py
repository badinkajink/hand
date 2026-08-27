#!/usr/bin/env python3
"""The reorientation primitive: what a reorient policy does in the OBJECT's frame.

    # extract from the two reference reorienters
    uv run --extra rl --extra gpu python scripts/reorient_primitive.py extract \
        --run results/rl/b33_20260702-1353-policyB_m05_reorient_ik10 --checkpoint model_270.pt \
        --label b33_m05 --out results/reorient_primitive/b33_m05.npz
    # compare them
    uv run python scripts/reorient_primitive.py compare \
        results/reorient_primitive/*.npz --out docs/experiments/<date>-primitive/

WHY. Every representation this program has used to describe a reorient is hand-specific: joint
angles, a CEM reference trajectory, a residual budget in joint space. That is why evaluating a
morphology costs a training run -- the description of the skill does not survive a change of
hand, so the skill has to be re-learned rather than re-targeted.

The object does not care which hand is holding it. In the OBJECT's own frame a reorient is a
schedule of contact points sweeping over its surface while a net wrench turns it. That
description is hand-free by construction, and if two policies on two DIFFERENT topologies
(m05's inline+thumb roller, perp's opposed-pair pivoter) trace the same schedule, then a single
object-frame primitive describes the skill and morphology only decides who can realise it.

The existing object-relative fingertip reference (`src/morphohand/rl/imitation.py`) is the same
intuition stopped one step short: it is indexed by CLOCK, so it is an open-loop demo that can
only be used as a shaping reward. Index the same data by the object's own PHASE (its alignment
with vertical) and it becomes a feedback law -- desired contact configuration as a function of
object state -- which is executable, on any hand, with no training.

WHAT IS RECORDED, all per step and per env:
  object pose/vel        -> phase, and the net applied wrench from rigid-body dynamics
  fingertip poses        -> contact schedule in object-surface coordinates
  palm pose              -> how much of the turn is WRIST, not fingers. Both reference
                            policies carry palm_rotation_residual_scale 0.3, so a fingertip-only
                            representation is only honest if the wrist share is measured and
                            small. This is the term that decides whether the primitive is
                            about fingers at all.
  per-finger force       -> which contacts are load-bearing at each phase (a tip that is near
                            the surface but at 0 N is not part of the schedule)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TIP_BODIES = ("thumb_tip", "index_tip", "middle_tip")
PALM_BODY = "palm_pose"


def extract(args) -> None:
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    import yaml

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner
    from morphohand.rl.env_cfg import MorphoHandEnvCfg, to_mjlab_cfg
    from morphohand.rl.ppo_config import PPOConfig
    from morphohand.rl.ppo_runner import build_runner_cfg

    run = args.run.resolve()
    with (run / "config.yaml").open() as f:
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
    kw["num_envs"] = args.n
    kw["episode_length_s"] = args.steps / 50.0 + 0.5
    # Terminations off: a truncated episode leaves the tail of the recording as stale repeats
    # of the terminal state, which would be resampled into the primitive as a fake plateau.
    kw["enable_lift_terminations"] = False

    cfg = MorphoHandEnvCfg(**kw)
    env = ManagerBasedRlEnv(cfg=to_mjlab_cfg(cfg), device="cuda:0", render_mode=None)
    wrapped = RslRlVecEnvWrapper(env)
    runner = ManipulationOnPolicyRunner(
        env=wrapped,
        train_cfg=dataclasses.asdict(build_runner_cfg(
            PPOConfig(num_envs=args.n), Path("/tmp/reoprim"), run_name="p")),
        log_dir="/tmp/reoprim/tb", device="cuda:0")
    ck = torch.load(str(run / "tensorboard" / args.checkpoint),
                    map_location="cpu", weights_only=False)
    runner.alg.actor.load_state_dict(ck["actor_state_dict"], strict=True)
    runner.alg.actor.eval()
    actor = runner.alg.actor

    # Optional Policy A prefix. b33 trained under the live-A reset -- a frozen a10 drove the
    # real lift over 0..onset and B's pre-onset steps were PPO-masked -- so B alone from step 0
    # is out of distribution on its own delivery, and its recorded schedule would be a
    # different policy's. The live-A checkpoint is a CLI flag, not a config field, so it
    # cannot be recovered from the run; it has to be named.
    actor_a = None
    if args.policy_a is not None:
        cka = torch.load(str(args.policy_a), map_location="cpu", weights_only=False)
        a_dim = int(next(cka["actor_state_dict"][k] for k in cka["actor_state_dict"]
                         if k.endswith("weight")).shape[1])
        # A throwaway env at A's own obs width, only so rsl_rl instantiates an actor of the
        # right shape (65 = no target-axis term). Same trick as rl_demo_handoff_continuous.
        kw_a = dict(kw)
        kw_a["enable_target_axis_reward"] = (a_dim == 66)
        kw_a["episode_length_s"] = 0.5
        env_a = ManagerBasedRlEnv(cfg=to_mjlab_cfg(MorphoHandEnvCfg(**kw_a)),
                                  device="cuda:0", render_mode=None)
        wrapped_a = RslRlVecEnvWrapper(env_a)
        ra = ManipulationOnPolicyRunner(
            env=wrapped_a,
            train_cfg=dataclasses.asdict(build_runner_cfg(
                PPOConfig(num_envs=args.n), Path("/tmp/reoprimA"), run_name="a")),
            log_dir="/tmp/reoprimA/tb", device="cuda:0")
        ra.alg.actor.load_state_dict(cka["actor_state_dict"], strict=True)
        ra.alg.actor.eval()
        actor_a = (ra.alg.actor, a_dim)
        env_a.close()

    robot = env.scene["robot"]
    obj = env.scene["cube"]
    tip_ids = [robot.body_names.index(b) for b in TIP_BODIES]
    palm_id = robot.body_names.index(PALM_BODY)

    T, N = args.steps, args.n
    rec = dict(
        obj_pose=np.zeros((T, N, 7), np.float32),
        obj_linvel=np.zeros((T, N, 3), np.float32),
        obj_angvel=np.zeros((T, N, 3), np.float32),
        tips_w=np.zeros((T, N, 3, 3), np.float32),
        palm_pose=np.zeros((T, N, 7), np.float32),
        force=np.zeros((T, N, 3), np.float32),
        joint_pos=np.zeros((T, N, robot.data.joint_pos.shape[-1]), np.float32),
    )

    obs_td, _ = wrapped.reset()
    sensor = "fingertip_cube_contact"
    with torch.no_grad():
        for s in range(T):
            obs = obs_td["actor"]
            if actor_a is not None and s < args.handoff_step:
                aa, a_dim = actor_a
                a = (aa.act_inference(obs[:, :a_dim]) if hasattr(aa, "act_inference")
                     else aa.mlp(obs[:, :a_dim]))
            else:
                a = actor(obs_td)
            obs_td, *_ = wrapped.step(a)
            rec["obj_pose"][s] = obj.data.root_link_pose_w.cpu().numpy()
            rec["obj_linvel"][s] = obj.data.root_link_lin_vel_w.cpu().numpy()
            rec["obj_angvel"][s] = obj.data.root_link_ang_vel_w.cpu().numpy()
            rec["tips_w"][s] = robot.data.body_link_pose_w[:, tip_ids, :3].cpu().numpy()
            rec["palm_pose"][s] = robot.data.body_link_pose_w[:, palm_id, :].cpu().numpy()
            rec["joint_pos"][s] = robot.data.joint_pos.cpu().numpy()
            f = env.scene.sensors[sensor].data.force
            if f is not None:
                mag = f.norm(dim=-1)
                if mag.dim() == 3:
                    mag = mag.amax(dim=-1)
                rec["force"][s, :, :mag.shape[1]] = mag[:, :3].cpu().numpy()

    # Object inertial properties, for the wrench. Read off the compiled model so the analysis
    # never has to assume the object is the same across scenes.
    import mujoco
    mm = mujoco.MjModel.from_xml_path(str(kw["frozen_scene_xml"]))
    bid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, cfg.object_body_name)
    gsize = np.zeros(3)
    for gi in range(mm.ngeom):
        if mm.geom_bodyid[gi] == bid:
            gsize = mm.geom_size[gi].copy()
            break

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, label=args.label, run=str(run), checkpoint=args.checkpoint,
        policy_a=str(args.policy_a or ""), handoff_step=int(args.handoff_step),
        dt=1.0 / 50.0, reorient_start_step=int(kw.get("reorient_start_step", 0)),
        obj_mass=float(mm.body_mass[bid]), obj_inertia=mm.body_inertia[bid].copy(),
        obj_radius=float(gsize[0]), obj_half_len=float(gsize[1]),
        tip_names=np.array(TIP_BODIES), **rec)
    env.close()

    cos = 1.0 - 2.0 * (rec["obj_pose"][..., 4] ** 2 + rec["obj_pose"][..., 5] ** 2)
    print(f"[{args.label}] wrote {out}  T={T} N={N}  "
          f"peak_cos={cos.max(axis=0).mean():+.3f}  final_cos={cos[-50:].mean():+.3f}  "
          f"final_z={rec['obj_pose'][-1, :, 2].mean():.4f}")




# ---------------------------------------------------------------------------
# Analysis. Pure numpy -- no GPU, no mjlab; runs on the recorded npz.
# ---------------------------------------------------------------------------

def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=-2)


def analyse(path: Path) -> dict:
    """Everything the primitive is made of, in the object's own frame.

    THE TILT FRAME. A cylinder is rotationally symmetric about its long axis, so an absolute
    azimuth in the object's body frame is not observable -- it is set by whatever yaw the object
    happened to spawn with. What IS observable, and is the natural datum for this task, is where
    UP is as seen by the shaft: project world +Z into the shaft's own cross-section and call that
    direction theta = 0. Then a fingertip's theta says whether it rides the upper or the lower
    side of the shaft, an axis-symmetry-invariant, hand-free statement of the contact geometry.
    It degenerates only when the shaft is vertical, which is where the reorient ends anyway.

    ROLL vs PIVOT falls straight out of the angular velocity in that same frame: the component
    ALONG the shaft is a roll (the shaft spinning in the fingers), the component across it is the
    tilt that the task actually asks for. Their ratio is the whole strategy in one number.
    """
    d = np.load(path, allow_pickle=False)
    pose, tips_w = d["obj_pose"], d["tips_w"]              # (T,N,7), (T,N,3,3)
    palm, force = d["palm_pose"], d["force"]               # (T,N,7), (T,N,3)
    w_w = d["obj_angvel"]                                  # (T,N,3) world
    T, N = pose.shape[:2]

    R = _quat_to_mat(pose[..., 3:7])                       # (T,N,3,3) body->world
    cos = np.clip(R[..., 2, 2], -1.0, 1.0)                 # object +Z . world +Z

    # tip positions in the object body frame
    rel = tips_w - pose[:, :, None, :3]                    # (T,N,3,3) world-offset
    tip_b = np.einsum("tnji,tnkj->tnki", R, rel)           # R^T @ rel

    # tilt-frame azimuth: world up expressed in the body frame, projected to the cross-section
    up_b = R[..., 2, :]                                    # (T,N,3) = R^T @ [0,0,1]
    phi_u = np.arctan2(up_b[..., 1], up_b[..., 0])         # (T,N)
    theta = np.arctan2(tip_b[..., 1], tip_b[..., 0]) - phi_u[..., None]
    theta = (theta + np.pi) % (2 * np.pi) - np.pi          # (T,N,3) wrapped
    radius = np.hypot(tip_b[..., 0], tip_b[..., 1])
    axial = tip_b[..., 2]

    # angular velocity in the body frame -> roll (along shaft) vs tilt (across it)
    w_b = np.einsum("tnji,tnj->tni", R, w_w)
    w_roll = w_b[..., 2]
    w_tilt = np.linalg.norm(w_b[..., :2], axis=-1)

    # how much of the turn is the WRIST: the object's tilt measured in the palm frame vs world.
    Rp = _quat_to_mat(palm[..., 3:7])
    zb_w = R[..., :, 2]                                    # shaft axis in world
    zb_p = np.einsum("tnji,tnj->tni", Rp, zb_w)            # shaft axis in the palm frame
    cos_palm = np.clip(zb_p[..., 2], -1.0, 1.0)

    # Only envs that actually did the task define the primitive. An env that dropped the
    # shaft still produces a contact schedule; it is just not a schedule for reorienting.
    onset = int(d["reorient_start_step"])
    good = (cos.max(axis=0) >= 0.8) & (pose[-1, :, 2] > 0.06) & (force[-1].sum(-1) > 0.5)
    if good.sum() == 0:
        good = cos.max(axis=0) >= 0.5

    # object pose in the PALM frame. The contact schedule says where on the shaft to press;
    # this says where the shaft is relative to the hand while you press there. Together they
    # are a complete, hand-free statement of the task -- and they are what a kinematic
    # feasibility test needs, because a design can fail either by not reaching the contact
    # point or by not being able to hold the shaft in that place at all.
    palm_pos, palm_quat = palm[..., :3], palm[..., 3:7]
    obj_in_palm_pos = np.einsum("tnji,tnj->tni", Rp, pose[..., :3] - palm_pos)
    obj_in_palm_axis = zb_p

    return dict(label=str(d["label"]), path=str(path), T=T, N=N, good=good, onset=onset,
                obj_in_palm_pos=obj_in_palm_pos, obj_in_palm_axis=obj_in_palm_axis,
                cos=cos, cos_palm=cos_palm, theta=theta, radius=radius, axial=axial,
                force=force, w_roll=w_roll, w_tilt=w_tilt, obj_z=pose[..., 2],
                obj_r=float(d["obj_radius"]), obj_h=float(d["obj_half_len"]),
                tip_names=[str(x) for x in d["tip_names"]])


def _resample_by_phase(a: np.ndarray, cos: np.ndarray, good: np.ndarray,
                       grid: np.ndarray, onset: int) -> np.ndarray:
    """Resample a per-step quantity onto a grid of PHASE (alignment cos).

    Phase, not time, is the whole point. Two hands that trace the same contact path at
    different speeds are the same primitive; indexed by clock they look like different ones,
    which is the flaw in the existing time-indexed fingertip reference. Uses each env's
    monotone rising segment from onset to its peak, so a policy that overshoots and falls back
    contributes only its outbound path.
    """
    T, N = cos.shape
    out = np.full((len(grid),) + ((a.shape[2],) if a.ndim == 3 else ()) + (N,), np.nan, np.float32)
    for e in np.nonzero(good)[0]:
        c = cos[onset:, e]
        pk = int(np.argmax(c))
        if pk < 3:
            continue
        cc = np.maximum.accumulate(c[:pk + 1])             # monotone outbound phase
        seg = a[onset:onset + pk + 1, e]
        keep = np.concatenate([[True], np.diff(cc) > 1e-6])
        if keep.sum() < 3:
            continue
        cc, seg = cc[keep], seg[keep]
        if a.ndim == 3:
            for k in range(a.shape[2]):
                out[:, k, e] = np.interp(grid, cc, seg[:, k], left=np.nan, right=np.nan)
        else:
            out[:, e] = np.interp(grid, cc, seg, left=np.nan, right=np.nan)
    return out


def compare(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [analyse(p) for p in args.npz]
    if args.onset is not None:
        for r in runs:
            r["onset"] = int(args.onset)
    grid = np.linspace(0.0, args.phase_max, args.phase_bins)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, prims = [], {}
    for r in runs:
        g, onset = r["good"], r["onset"]
        th = _resample_by_phase(r["theta"], r["cos"], g, grid, onset)      # (P,3,N)
        ax = _resample_by_phase(r["axial"], r["cos"], g, grid, onset)
        rd = _resample_by_phase(r["radius"], r["cos"], g, grid, onset)
        fo = _resample_by_phase(r["force"], r["cos"], g, grid, onset)
        wr = _resample_by_phase(r["w_roll"], r["cos"], g, grid, onset)     # (P,N)
        wt = _resample_by_phase(r["w_tilt"], r["cos"], g, grid, onset)

        # circular mean over envs for the azimuth; plain mean for the rest
        th_m = np.angle(np.nanmean(np.exp(1j * th), axis=-1))
        th_sd = np.sqrt(-2.0 * np.log(np.clip(np.abs(np.nanmean(np.exp(1j * th), axis=-1)), 1e-9, 1)))
        prims[r["label"]] = dict(grid=grid, theta=th_m, theta_sd=th_sd,
                                 axial=np.nanmean(ax, -1), radius=np.nanmean(rd, -1),
                                 force=np.nanmean(fo, -1), w_roll=np.nanmean(wr, -1),
                                 w_tilt=np.nanmean(wt, -1))

        # headline scalars, all hand-free
        cos = r["cos"]
        sel = slice(onset, None)
        # PER ENV, then averaged. Averaging the SIGNED roll over envs first lets opposite-sign
        # draws cancel and reports a roller as a pivoter -- it read m05 at 0.30 when it is 1.9.
        aw, at_ = np.abs(r["w_roll"][:, g]), r["w_tilt"][:, g]
        cpk = r["cos"][:, g].argmax(axis=0)
        rat = [aw[onset:pk, e].mean() / max(at_[onset:pk, e].mean(), 1e-9)
               for e, pk in enumerate(cpk) if pk > onset + 5]
        rollness = float(np.mean(rat)) if rat else float("nan")
        # wrist share: how much of the achieved tilt is already there in the PALM frame.
        # If the object never tilts relative to the palm, the wrist did the reorient.
        d_world = float(np.nanmean(cos[sel][:, g].max(0) - cos[onset, g]))
        d_palm = float(np.nanmean(r["cos_palm"][sel][:, g].max(0) - r["cos_palm"][onset, g]))
        summary.append(dict(
            label=r["label"], n_good=int(g.sum()), n=int(r["N"]),
            peak_cos=float(np.nanmean(cos[:, g].max(0))),
            final_cos=float(np.nanmean(cos[-50:][:, g])),
            roll_over_tilt=rollness,
            tilt_gain_world=d_world, tilt_gain_palm=d_palm,
            wrist_share=float(1.0 - d_palm / d_world) if abs(d_world) > 1e-6 else float("nan"),
            mean_contacts=float(np.nanmean((r["force"][sel][:, g] > 0.2).sum(-1))),
            mean_force_N=float(np.nanmean(r["force"][sel][:, g].sum(-1))),
        ))

    labels = [r["label"] for r in runs]
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.5))
    colors = plt.cm.tab10(np.arange(len(runs)))
    tipn = runs[0]["tip_names"]
    ls = ["-", "--", ":"]

    for i, lab in enumerate(labels):
        P = prims[lab]
        for k in range(3):
            axs[0, 0].plot(grid, np.degrees(P["theta"][:, k]), ls[k], color=colors[i],
                           label=f"{lab} {tipn[k].split('_')[0]}")
            axs[0, 1].plot(grid, P["axial"][:, k] * 1000, ls[k], color=colors[i])
            axs[1, 0].plot(grid, P["force"][:, k], ls[k], color=colors[i])
        axs[0, 2].plot(grid, P["w_roll"], "-", color=colors[i], label=f"{lab} roll")
        axs[0, 2].plot(grid, P["w_tilt"], "--", color=colors[i], label=f"{lab} tilt")
        # per-finger, LOADED only: averaging an idle finger's stand-off into this reads as
        # "the contacts sit 35 mm from a 12.5 mm shaft", which is not a contact at all
        for k in range(3):
            rr = np.where(P["force"][:, k] > 0.5, P["radius"][:, k], np.nan)
            axs[1, 1].plot(grid, rr * 1000, ls[k], color=colors[i],
                           label=f"{lab} {tipn[k].split('_')[0]}")
        r = runs[i]
        axs[1, 2].plot(r["cos"][:, r["good"]].mean(-1), color=colors[i], label=f"{lab} world")
        axs[1, 2].plot(r["cos_palm"][:, r["good"]].mean(-1), "--", color=colors[i],
                       label=f"{lab} in palm")

    axs[0, 0].set(xlabel="phase (alignment cos)", ylabel="tilt-frame azimuth (deg)",
                  title="WHERE each tip rides on the shaft\n(0 = up side, ±180 = down side)")
    axs[0, 0].axhline(0, color="k", lw=0.5)
    axs[0, 0].legend(fontsize=6, ncol=2)
    axs[0, 1].set(xlabel="phase (alignment cos)", ylabel="axial position (mm)",
                  title="HOW FAR ALONG the shaft (0 = centre)")
    axs[0, 1].axhline(0, color="k", lw=0.5)
    axs[0, 2].set(xlabel="phase (alignment cos)", ylabel="body-frame ang. vel (rad/s)",
                  title="ROLL (along shaft) vs TILT (across it)")
    axs[0, 2].legend(fontsize=7)
    axs[1, 0].set(xlabel="phase (alignment cos)", ylabel="contact force (N)",
                  title="WHICH tips are load-bearing, per phase")
    axs[1, 1].set(xlabel="phase (alignment cos)", ylabel="contact radius (mm)",
                  title="tip stand-off from the shaft axis\n(loaded fingers only; dotted = surface)")
    axs[1, 1].axhline(runs[0]["obj_r"] * 1000, color="k", lw=0.5, ls=":")
    axs[1, 1].legend(fontsize=6, ncol=2)
    axs[1, 2].set(xlabel="step", ylabel="alignment cos",
                  title="world vs palm-frame tilt\n(gap = the wrist's share)")
    axs[1, 2].legend(fontsize=7)
    for a in axs.ravel():
        a.grid(alpha=0.3)
    fig.suptitle("The reorientation primitive in the object's own frame", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "primitive_compare.png", dpi=130)

    # One executable primitive file per policy: the phase-indexed contact schedule in the
    # tilt frame, plus which fingers were load-bearing. This is the whole representation --
    # no joint angles, no link lengths, nothing that names a hand.
    for r, lab in zip(runs, labels):
        P = prims[lab]
        g, onset = r["good"], r["onset"]
        fo = _resample_by_phase(r["force"], r["cos"], g, grid, onset)
        loaded = (np.nanmean(fo, -1) > 0.5)
        # Keep only the phase span the policy actually covered. Outside it every env
        # contributed NaN, and a table with NaN at its ends silently poisons np.interp for
        # EVERY query -- which is exactly how the first controller run emitted NaN ctrl and
        # read as "the hand cannot do it".
        ok = np.isfinite(P["theta"]).all(-1) & np.isfinite(P["axial"]).all(-1)
        if not ok.any():
            raise ValueError(f"{lab}: no phase bin has data")
        i0, i1 = int(np.argmax(ok)), len(ok) - int(np.argmax(ok[::-1]))
        sl = slice(i0, i1)
        oip = _resample_by_phase(r["obj_in_palm_pos"], r["cos"], g, grid, onset)
        oia = _resample_by_phase(r["obj_in_palm_axis"], r["cos"], g, grid, onset)
        np.savez(out_dir / f"{lab}__primitive.npz", grid=grid[sl], theta=P["theta"][sl],
                 obj_in_palm_pos=np.nanmean(oip, -1)[sl],
                 obj_in_palm_axis=np.nanmean(oia, -1)[sl],
                 theta_sd=P["theta_sd"][sl], axial=P["axial"][sl], radius=P["radius"][sl],
                 force=P["force"][sl], loaded=loaded[sl], label=lab,
                 obj_radius=r["obj_r"], obj_half_len=r["obj_h"],
                 tip_names=np.array(r["tip_names"]))

    (out_dir / "primitive_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out_dir / "primitive_curves.npz",
                        **{f"{k}__{kk}": vv for k, v in prims.items() for kk, vv in v.items()})

    w = max(len(s_["label"]) for s_ in summary) + 1
    print(f"{'policy':{w}} {'good':>5} {'peak':>6} {'final':>6} {'roll/tilt':>9} "
          f"{'wrist%':>7} {'contacts':>8} {'force N':>8}")
    for s_ in summary:
        print(f"{s_['label']:{w}} {s_['n_good']:>2}/{s_['n']:<2} {s_['peak_cos']:>+6.3f} "
              f"{s_['final_cos']:>+6.3f} {s_['roll_over_tilt']:>9.2f} "
              f"{100 * s_['wrist_share']:>6.1f}% {s_['mean_contacts']:>8.2f} "
              f"{s_['mean_force_N']:>8.2f}")
    print(f"\nwrote {out_dir / 'primitive_compare.png'}")




# ---------------------------------------------------------------------------
# Execution. The primitive as a CONTROLLER, in CPU MuJoCo, on any hand.
# ---------------------------------------------------------------------------

FINGERS = ("thumb", "index", "middle")


def _ease_out_quad(t: float) -> float:
    return 1.0 - (1.0 - t) ** 2


def execute(args) -> None:
    """Run a recorded primitive as a closed-loop controller on an arbitrary hand.

    Nothing here knows what hand it is driving. The primitive says where on the SHAFT each
    load-bearing fingertip should sit as a function of the shaft's own alignment; the only
    hand-specific step is the tip Jacobian that turns a Cartesian tip error into joint
    targets. If a design can realise the schedule it reorients; if it cannot, the IK residual
    says so directly, and no training run was spent finding that out.

    The set-point is read at phase + `--lookahead`, not at the current phase: a controller
    that tracks where the demonstration already is has nothing to push against. The lookahead
    is what converts a demonstration into a drive.
    """
    import mujoco

    from morphohand.sampling.morphology import FINGER_ACTUATOR_NAMES

    prim = np.load(args.primitive, allow_pickle=False)
    grid = prim["grid"]
    m = mujoco.MjModel.from_xml_path(str(args.scene))
    d = mujoco.MjData(m)

    def aid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    fing_act = [aid(a) for a in FINGER_ACTUATOR_NAMES]
    palm_act = [aid(f"a_palm_{n}") for n in ("px", "py", "pz", "rx", "ry", "rz")]
    jnt_names = [a[2:] for a in FINGER_ACTUATOR_NAMES]
    dofadr = {}
    qposadr = {}
    for jn in jnt_names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        dofadr[jn] = int(m.jnt_dofadr[jid])
        qposadr[jn] = int(m.jnt_qposadr[jid])
    tip_bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{f}_tip") for f in FINGERS]
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args.object_body)

    # pad radius per fingertip, off the model -- the primitive names a point on the SHAFT
    # surface, and each hand stands its own pad off that surface by its own radius.
    pad_r = []
    for b in tip_bid:
        rr = [float(m.geom_size[gi][0]) for gi in range(m.ngeom) if m.geom_bodyid[gi] == b]
        pad_r.append(max(rr) if rr else 0.005)
    Rs = float(prim["obj_radius"])

    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    mujoco.mj_resetDataKeyframe(m, d, kid)
    if args.object_shift:
        # Slide the shaft along its OWN long axis before the grasp, so the same scripted grasp
        # lands at a different axial offset from the centre of mass. This is the whole
        # pendulum hypothesis made into a knob: nothing about the hand or the policy changes,
        # only WHERE along the tool it is held.
        mujoco.mj_forward(m, d)
        axis = d.xmat[obj_bid].reshape(3, 3)[:, 2]
        jadr = int(m.jnt_qposadr[int(m.body_jntadr[obj_bid])])
        d.qpos[jadr:jadr + 3] += args.object_shift * axis
        mujoco.mj_forward(m, d)
    open_ctrl = np.array([d.ctrl[a] for a in fing_act], float)
    palm0 = np.array([d.ctrl[a] for a in palm_act], float)
    closed_ctrl = (np.load(args.closed_ctrl)["best_finger_ctrl"].reshape(-1).astype(float)
                   if args.closed_ctrl else open_ctrl.copy())
    # Grip tightness as a scalar between the open pose and the CEM grasp. These are position
    # servos, so how far past the shaft's surface the set-point sits IS the squeeze; a grasp
    # that clamps too hard cannot let the shaft pivot, however well placed it is.
    closed_ctrl = open_ctrl + args.grip_scale * (closed_ctrl - open_ctrl)

    sub = max(1, int(round(args.control_dt / m.opt.timestep)))
    lo = np.array([m.actuator_ctrlrange[a][0] for a in fing_act])
    hi = np.array([m.actuator_ctrlrange[a][1] for a in fing_act])

    jacp = np.zeros((3, m.nv))
    rec = {k: [] for k in ("cos", "z", "force", "ikres", "phase")}
    tracking = False

    for step in range(args.steps):
        t = min(1.0, (step + 1) / args.close_steps)
        anchor = open_ctrl + (closed_ctrl - open_ctrl) * _ease_out_quad(t)
        lift = float(np.clip((step - args.settle_steps + 1) / args.lift_ramp_steps, 0.0, 1.0))
        for k, a in enumerate(palm_act):
            d.ctrl[a] = palm0[k] + (args.lift_delta if k == 2 else 0.0) * lift

        # per-finger contact force as measured right now -- the feedback for the force half of
        # the schedule. A contact schedule is a position AND a load; a tracker that only places
        # the tips can hold a shaft but cannot press one into turning.
        fnow = np.zeros(3)
        for ci in range(d.ncon):
            c = d.contact[ci]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for fi_, tb_ in enumerate(tip_bid):
                if (b1 == tb_ and b2 == obj_bid) or (b2 == tb_ and b1 == obj_bid):
                    ft_ = np.zeros(6)
                    mujoco.mj_contactForce(m, d, ci, ft_)
                    fnow[fi_] += float(np.linalg.norm(ft_[:3]))

        cmd = anchor.copy()
        ikres = 0.0
        if step >= args.prim_start_step:
            R = _quat_to_mat(d.xquat[obj_bid][None])[0]
            p_obj = d.xpos[obj_bid].copy()
            cos = float(np.clip(R[2, 2], -1, 1))
            up_b = R[2, :]
            phi_u = float(np.arctan2(up_b[1], up_b[0]))
            tgt_phase = float(np.clip(cos + args.lookahead, grid[0], grid[-1]))
            if not np.isfinite(tgt_phase):
                raise RuntimeError("non-finite phase; the sim has diverged")
            if cos >= args.prim_start_cos:
                tracking = True
            if tracking:
                for fi in range(3):
                    if not bool(prim["loaded"][:, fi].any()):
                        continue
                    th = float(np.interp(tgt_phase, grid, np.unwrap(prim["theta"][:, fi])))
                    sax = float(np.interp(tgt_phase, grid, prim["axial"][:, fi]))
                    a_ = phi_u + th
                    # target tip CENTRE: the schedule's point on the shaft surface, stood off
                    # by this hand's own pad radius along the surface normal
                    # Force tracking as impedance: a finger that is under-loaded gets a target
                    # pushed INTO the shaft, an over-loaded one gets it pulled out. Same IK, no
                    # second controller -- the load schedule becomes a radial offset.
                    fstar = float(np.interp(tgt_phase, grid, prim["force"][:, fi]))
                    rad = Rs + pad_r[fi] - args.force_gain * (fstar - fnow[fi])
                    rad = float(np.clip(rad, args.min_radius, Rs + pad_r[fi] + 0.01))
                    p_b = np.array([rad * np.cos(a_), rad * np.sin(a_), sax])
                    p_t = p_obj + R @ p_b
                    dx = p_t - d.xpos[tip_bid[fi]]
                    ikres += float(np.linalg.norm(dx))
                    mujoco.mj_jacBody(m, d, jacp, None, tip_bid[fi])
                    cols = [dofadr[f"{FINGERS[fi]}_{j}"] for j in ("yaw", "mcp", "pip")]
                    J = jacp[:, cols]
                    if not np.all(np.isfinite(dx)):
                        raise RuntimeError(f"non-finite tip target for {FINGERS[fi]}")
                    dq = J.T @ np.linalg.solve(J @ J.T + (args.damping ** 2) * np.eye(3), dx)
                    # A RESIDUAL on the grasp anchor, never a replacement. These are position
                    # servos: the grip force is the commanded-minus-actual joint error that the
                    # squeezing anchor supplies. Commanding the joints to put the tip exactly
                    # ON the surface sets that error to ~0 and the hand lets go -- which is
                    # what the first version of this did, and it dropped the shaft every time.
                    # It is also the action space the trained policies use, so a primitive
                    # executed this way is directly comparable to a learned residual.
                    k0 = fi * 3
                    cmd[k0:k0 + 3] = anchor[k0:k0 + 3] + np.clip(
                        args.ik_gain * dq, -args.residual_budget, args.residual_budget)
        for k, a in enumerate(fing_act):
            d.ctrl[a] = float(np.clip(cmd[k], lo[k], hi[k]))
        for _ in range(sub):
            mujoco.mj_step(m, d)

        R = _quat_to_mat(d.xquat[obj_bid][None])[0]
        rec["cos"].append(float(np.clip(R[2, 2], -1, 1)))
        rec["z"].append(float(d.xpos[obj_bid][2]))
        rec["phase"].append(float(np.clip(R[2, 2], -1, 1)))
        rec["ikres"].append(ikres)
        f = np.zeros(3)
        for ci in range(d.ncon):
            c = d.contact[ci]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for fi, tb in enumerate(tip_bid):
                if (b1 == tb and b2 == obj_bid) or (b2 == tb and b1 == obj_bid):
                    ft = np.zeros(6)
                    mujoco.mj_contactForce(m, d, ci, ft)
                    f[fi] += float(np.linalg.norm(ft[:3]))
        rec["force"].append(f)
        # where the LOADED contacts sit along the shaft, in the object's own frame -- the
        # quantity the pendulum story is about
        Rn = d.xmat[obj_bid].reshape(3, 3)
        sax_now = []
        for fi in range(3):
            if f[fi] > 0.2:
                sax_now.append(float((Rn.T @ (d.xpos[tip_bid[fi]] - d.xpos[obj_bid]))[2]))
        rec.setdefault("axial", []).append(np.mean(sax_now) if sax_now else np.nan)

    cos = np.array(rec["cos"])
    z = np.array(rec["z"])
    fo = np.array(rec["force"])
    tail = slice(-args.tail_steps, None)
    print(f"{args.label:26s} peak_cos {cos.max():+.3f}  final_cos {cos[tail].mean():+.3f}  "
          f"final_z {z[tail].mean():.4f}  force {fo[tail].sum(-1).mean():5.2f} N  "
          f"contacts {(fo[tail] > 0.2).sum(-1).mean():.2f}  "
          f"s_contact {np.nanmean(np.array(rec['axial'])[tail]) * 1000:+6.1f}mm  "
          f"ik_res {np.mean(rec['ikres'][args.prim_start_step:]) * 1000:5.1f} mm")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(dict(
            label=args.label, scene=str(args.scene), primitive=str(args.primitive),
            peak_cos=float(cos.max()), final_cos=float(cos[tail].mean()),
            final_z=float(z[tail].mean()), final_force=float(fo[tail].sum(-1).mean()),
            contacts=float((fo[tail] > 0.2).sum(-1).mean()),
            ik_residual_mm=float(np.mean(rec["ikres"][args.prim_start_step:]) * 1000),
            cos=cos.tolist(), z=z.tolist()), indent=2))




def feasibility(args) -> None:
    """Can this hand realise the primitive? Pure kinematics -- no physics, no policy.

    THE POINT. Scoring a morphology currently costs a training run, and the answer is
    seed-dominated on top of that. But a large part of what a design can or cannot do is
    settled before any dynamics: the primitive says where the shaft sits relative to the palm
    and where on its surface each loaded finger must press, and a hand either reaches those
    points or it does not. That question has one answer, it takes seconds, and nothing about
    it can be a lucky draw.

    This is also the repair of the metric that failed. UHAS's authority score asked how much
    wrench a grasp could resist -- a task-blind question, and it correlated -0.03 with goal
    approach. Here the task is not assumed, it is READ OFF a policy that does it, so what gets
    scored is the hand's ability to serve THIS trajectory rather than its generic strength.

    Reported per design: the mean tip residual over the schedule, and the fraction of
    (phase, finger) demands met within tolerance. Both are reachability, not success -- a hand
    that clears this can still fail dynamically, but a hand that fails it cannot succeed.
    """
    import mujoco

    from morphohand.sampling.morphology import FINGER_ACTUATOR_NAMES

    prim = np.load(args.primitive, allow_pickle=False)
    grid = prim["grid"]
    Rs = float(prim["obj_radius"])
    m = mujoco.MjModel.from_xml_path(str(args.scene))
    d = mujoco.MjData(m)

    def aid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    fing_act = [aid(a) for a in FINGER_ACTUATOR_NAMES]
    jnames = [a[2:] for a in FINGER_ACTUATOR_NAMES]
    qadr = {j: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)])
            for j in jnames}
    dadr = {j: int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)])
            for j in jnames}
    jrange = {j: m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)].copy()
              for j in jnames}
    tip_bid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{f}_tip") for f in FINGERS]
    palm_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PALM_BODY)
    obj_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, args.object_body)
    obj_qadr = int(m.jnt_qposadr[int(m.body_jntadr[obj_bid])])
    pad_r = []
    for b in tip_bid:
        rr = [float(m.geom_size[gi][0]) for gi in range(m.ngeom) if m.geom_bodyid[gi] == b]
        pad_r.append(max(rr) if rr else 0.005)

    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    mujoco.mj_resetDataKeyframe(m, d, kid)
    if args.closed_ctrl:
        cc = np.load(args.closed_ctrl)["best_finger_ctrl"].reshape(-1).astype(float)
        for k, j in enumerate(jnames):
            d.qpos[qadr[j]] = cc[k]
    mujoco.mj_forward(m, d)
    q_grasp = {j: float(d.qpos[qadr[j]]) for j in jnames}

    jacp = np.zeros((3, m.nv))
    res = np.full((len(grid), 3), np.nan)
    sat = np.zeros((len(grid), 3))
    loaded_any = prim["loaded"].any(axis=0)

    for pi in range(len(grid)):
        # place the shaft where the primitive says it sits relative to the palm
        mujoco.mj_resetDataKeyframe(m, d, kid)
        for k, j in enumerate(jnames):
            d.qpos[qadr[j]] = q_grasp[j]
        mujoco.mj_forward(m, d)
        Rp = d.xmat[palm_bid].reshape(3, 3)
        p_palm = d.xpos[palm_bid].copy()
        p_obj = p_palm + Rp @ prim["obj_in_palm_pos"][pi]
        zb = Rp @ prim["obj_in_palm_axis"][pi]
        zb = zb / max(np.linalg.norm(zb), 1e-9)
        # any completion of the frame is valid: the shaft is a cylinder, and the tilt-frame
        # azimuth is measured from world-up, so a spin about the shaft cancels out exactly.
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, zb)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        xb = np.cross(tmp, zb)
        xb /= max(np.linalg.norm(xb), 1e-9)
        R = np.stack([xb, np.cross(zb, xb), zb], axis=1)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.reshape(-1))
        d.qpos[obj_qadr:obj_qadr + 3] = p_obj
        d.qpos[obj_qadr + 3:obj_qadr + 7] = q
        mujoco.mj_forward(m, d)

        up_b = R[2, :]
        phi_u = float(np.arctan2(up_b[1], up_b[0]))
        for fi in range(3):
            if not loaded_any[fi]:
                continue
            a_ = phi_u + float(prim["theta"][pi, fi])
            rad = Rs + pad_r[fi]
            p_t = p_obj + R @ np.array([rad * np.cos(a_), rad * np.sin(a_),
                                        float(prim["axial"][pi, fi])])
            cols = [dadr[f"{FINGERS[fi]}_{j}"] for j in ("yaw", "mcp", "pip")]
            adrs = [qadr[f"{FINGERS[fi]}_{j}"] for j in ("yaw", "mcp", "pip")]
            rngs = [jrange[f"{FINGERS[fi]}_{j}"] for j in ("yaw", "mcp", "pip")]
            for _ in range(args.ik_iters):
                mujoco.mj_forward(m, d)
                dx = p_t - d.xpos[tip_bid[fi]]
                if np.linalg.norm(dx) < 1e-4:
                    break
                mujoco.mj_jacBody(m, d, jacp, None, tip_bid[fi])
                J = jacp[:, cols]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), dx)
                for k, ad in enumerate(adrs):
                    lo_, hi_ = rngs[k]
                    v = d.qpos[ad] + dq[k]
                    d.qpos[ad] = float(np.clip(v, lo_, hi_) if hi_ > lo_ else v)
            mujoco.mj_forward(m, d)
            res[pi, fi] = float(np.linalg.norm(p_t - d.xpos[tip_bid[fi]]))
            sat[pi, fi] = float(np.mean([
                1.0 if (rngs[k][1] > rngs[k][0] and
                        min(abs(d.qpos[ad] - rngs[k][0]), abs(d.qpos[ad] - rngs[k][1])) < 1e-3)
                else 0.0 for k, ad in enumerate(adrs)]))

    r = res[:, loaded_any]
    met = float(np.mean(r < args.tol))
    print(f"{args.label:22s} mean_residual {np.nanmean(r) * 1000:6.2f} mm   "
          f"p90 {np.nanpercentile(r, 90) * 1000:6.2f} mm   "
          f"met(<{args.tol * 1000:.0f}mm) {100 * met:5.1f}%   "
          f"limit_saturation {100 * float(np.mean(sat[:, loaded_any])):5.1f}%")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(dict(
            label=args.label, scene=str(args.scene), primitive=str(args.primitive),
            mean_residual_mm=float(np.nanmean(r) * 1000),
            p90_residual_mm=float(np.nanpercentile(r, 90) * 1000),
            met_frac=met, saturation=float(np.mean(sat[:, loaded_any])),
            per_phase_residual_mm=(res * 1000).tolist(), grid=grid.tolist()), indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--label", required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--n", type=int, default=32)
    e.add_argument("--steps", type=int, default=300)
    e.add_argument("--policy-a", type=Path, default=None,
                   help="drive steps < --handoff-step with this Policy A checkpoint "
                        "(reproduces the live-A reset b33 was trained under)")
    e.add_argument("--handoff-step", type=int, default=58)
    e.set_defaults(func=extract)

    c = sub.add_parser("compare")
    c.add_argument("npz", nargs="+", type=Path)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--phase-bins", type=int, default=60)
    c.add_argument("--phase-max", type=float, default=0.95)
    c.add_argument("--onset", type=int, default=None,
                   help="override the reorient onset. 0 covers the WHOLE rollout, including "
                        "the part Policy A does against the floor -- which on m05 is most of "
                        "the alignment gain, and is invisible if you start at B's onset.")
    c.set_defaults(func=compare)

    x = sub.add_parser("execute")
    x.add_argument("--scene", type=Path, required=True)
    x.add_argument("--primitive", type=Path, required=True)
    x.add_argument("--label", default="exec")
    x.add_argument("--keyframe", default="open_ik")
    x.add_argument("--closed-ctrl", type=Path, default=None,
                   help="best_rollout.npz whose best_finger_ctrl is the grasp anchor")
    x.add_argument("--object-body", default="screwdriver_medium")
    x.add_argument("--steps", type=int, default=400)
    x.add_argument("--control-dt", type=float, default=0.02)
    x.add_argument("--close-steps", type=int, default=8,
                   help="control steps to close the fingers (env: finger_close_sim_steps/decimation)")
    x.add_argument("--settle-steps", type=int, default=24,
                   help="control step the palm lift starts (env: settle_steps/decimation)")
    x.add_argument("--lift-ramp-steps", type=int, default=8)
    x.add_argument("--lift-delta", type=float, default=0.10)
    x.add_argument("--prim-start-step", type=int, default=40)
    x.add_argument("--prim-start-cos", type=float, default=-1.0,
                   help="only start tracking once the shaft is at least this aligned")
    x.add_argument("--lookahead", type=float, default=0.05,
                   help="read the schedule this far AHEAD in phase -- the drive term")
    x.add_argument("--ik-gain", type=float, default=0.5)
    x.add_argument("--damping", type=float, default=0.02)
    x.add_argument("--residual-budget", type=float, default=0.5,
                   help="rad, matching the trained policies' finger_residual_scale")
    x.add_argument("--force-gain", type=float, default=0.0,
                   help="m of radial target offset per N of contact-force error (0 = position "
                        "tracking only). ~0.0005 gives 0.5 mm per N.")
    x.add_argument("--min-radius", type=float, default=0.004,
                   help="floor on the commanded tip radius, so a force-starved finger cannot be "
                        "driven through the shaft's axis")
    x.add_argument("--object-shift", type=float, default=0.0,
                   help="m to slide the object along its own long axis before the grasp")
    x.add_argument("--grip-scale", type=float, default=1.0,
                   help="1.0 = the CEM grasp set-point; <1 relaxes it toward the open pose")
    x.add_argument("--tail-steps", type=int, default=50)
    x.add_argument("--json-out", type=Path, default=None)
    x.set_defaults(func=execute)

    fz = sub.add_parser("feasibility")
    fz.add_argument("--scene", type=Path, required=True)
    fz.add_argument("--primitive", type=Path, required=True)
    fz.add_argument("--closed-ctrl", type=Path, default=None)
    fz.add_argument("--label", default="design")
    fz.add_argument("--keyframe", default="open_ik")
    fz.add_argument("--object-body", default="screwdriver_medium")
    fz.add_argument("--ik-iters", type=int, default=60)
    fz.add_argument("--tol", type=float, default=0.005)
    fz.add_argument("--json-out", type=Path, default=None)
    fz.set_defaults(func=feasibility)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
