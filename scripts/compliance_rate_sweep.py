"""Compliance-robustness sweep, RATE edition — N batched rollouts per point. (2026-07-10)

Why: the n=1 deterministic sweep (`compliance_robustness_sweep.py`) mis-scores marginal
policies — probing showed DR-A at soft is a knife-edge grasp that Warp solver noise decides
(~40% success across 32 identical mean-action envs), which n=1 read as a hard FAIL. This sweep
runs the SAME continuous A->B handoff (hard switch at step 40, deterministic mean actions —
the deploy condition; env-to-env spread comes from the solver) in a 32-env batch per
(policy pair, solimp level) and reports:

  hold-rate      P(post-handoff min-z > 0.05)
  reorient-rate  P(held AND tail-50 held-cos > 0.7)
  cos p50|held   median tail cos among held envs

Success bar for compliance DR: hold-rate ~1 AND reorient-rate high + FLAT across the band.

Run: MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/compliance_rate_sweep.py
Resumable (per-point JSON). Writes docs/experiments/COMPLIANCE_RATE.{json,txt}.
"""
from __future__ import annotations
import gc
import json
import time
from pathlib import Path

from morphohand.studies import runlib
from morphohand.studies.runlib import ROOT, RESULTS_RL as RL, final_ckpt, latest_run

# reuse the deterministic sweep's scene generator + levels so points are comparable
from compliance_robustness_sweep import COMPLIANCES, make_scene_dir

MORPH = ROOT / "results/phase1/landscape/m05_ik_cem"
JSON_P = ROOT / "docs/experiments/COMPLIANCE_RATE.json"
TXT_P = ROOT / "docs/experiments/COMPLIANCE_RATE.txt"
N_ENVS = 32
TOTAL_STEPS = 240
HANDOFF = 40
TAIL = 50


def policies():
    a10 = RL / "a10_20260702-1256-policyA_m05_ik10/tensorboard/model_609.pt"
    b33 = RL / "b33_20260702-1353-policyB_m05_reorient_ik10/tensorboard/model_270.pt"
    out = [("soft_b33", a10, b33)]
    a_cdr = final_ckpt(latest_run("*policyA_m05_cdr"))          # round-1 A (narrow band)
    b_cdr = final_ckpt(latest_run("*policyB_m05_cdr_imit_k1"))
    if a_cdr and b_cdr:
        out.append(("cdr1_imitB", a_cdr, b_cdr))
    a2 = final_ckpt(latest_run("*policyA_m05_cdr2"))            # widened band
    bf = final_ckpt(latest_run("*policyB_m05_cdr2_imitfloor_k0"))
    if a2 and bf:
        out.append(("cdr2_imitfloor", a2, bf))
    if a2:
        out.append(("cdr2A_b33", a2, b33))                      # isolate A's contribution
    return [(n, a, b) for n, a, b in out if a and b and a.exists() and b.exists()]


def rate_point(a_ck: Path, b_ck: Path, scene_dir: Path, work: Path):
    """One batched A->B handoff rollout; returns rate metrics. Fresh envs per call."""
    import numpy as np
    import torch
    from morphohand.rl import deploy

    bfc = tuple(float(v) for v in
                np.load(scene_dir / "best_rollout.npz")["best_finger_ctrl"].reshape(-1))
    keyframe = json.loads((scene_dir / "summary.json").read_text()).get(
        "keyframe", "open_short_manual")
    frozen = scene_dir / "frozen_scene.xml"

    a_obs_dim = deploy.ckpt_obs_dim(a_ck)
    cfg_a = deploy.make_env_cfg(frozen, keyframe, scene_dir, bfc,
                                enable_target_axis=(a_obs_dim == 66), num_steps=10,
                                open_finger_from_keyframe=True)
    env_a, _, actor_a = deploy.build_actor(cfg_a, a_ck, work)
    env_a.close()

    cfg_b = deploy.make_env_cfg(frozen, keyframe, scene_dir, bfc,
                                enable_target_axis=True, num_steps=TOTAL_STEPS,
                                open_finger_from_keyframe=True)
    cfg_b.num_envs = N_ENVS
    env, wrapped, actor_b = deploy.build_actor(cfg_b, b_ck, work)

    obs_td, _ = wrapped.reset()
    obj = env.scene["cube"]
    minz_post = torch.full((N_ENVS,), 1e9, device=env.device)
    cos_hist = []
    with torch.no_grad():
        for step in range(TOTAL_STEPS):
            if step < HANDOFF:
                a = deploy.act(actor_a, obs_td["actor"][:, :a_obs_dim])
            else:
                a = deploy.act_b(actor_b, obs_td, stochastic=False)
            obs_td, *_ = wrapped.step(a)
            q = obj.data.root_link_quat_w        # (N,4) wxyz
            cos = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
            z = obj.data.root_link_pos_w[:, 2]
            if step >= HANDOFF:
                minz_post = torch.minimum(minz_post, z)
            cos_hist.append(cos.clone())

    tail_cos = torch.stack(cos_hist[-TAIL:]).mean(dim=0)
    held = minz_post > 0.05
    reor = held & (tail_cos > 0.7)
    held_cos = tail_cos[held]
    res = dict(
        n=N_ENVS,
        hold_rate=round(float(held.float().mean()), 3),
        reorient_rate=round(float(reor.float().mean()), 3),
        cos_p50_held=(round(float(held_cos.median()), 3) if held.any() else None),
        minz_p50=round(float(minz_post.median()), 4),
    )
    env.close()
    del env, wrapped, actor_a, actor_b
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    store = runlib.RecordStore(JSON_P, key_field="key")
    report = runlib.TxtReport(
        TXT_P, f"# compliance RATE sweep {time.strftime('%Y-%m-%d %H:%M')}  "
               f"n={N_ENVS}/point, deterministic actions, solver-noise spread\n"
               f"# hold-rate / reorient-rate(cos>0.7) / median held cos\n")
    work = ROOT / "logs" / "_rate_sweep_tmp"
    work.mkdir(parents=True, exist_ok=True)
    pols = policies()
    print("pairs:", [p[0] for p in pols])
    for dmin, dmax, label in COMPLIANCES:
        scene = make_scene_dir(dmin, dmax, label)
        for name, a_ck, b_ck in pols:
            key = f"{name}@{label}"
            if key in store:
                print(f"[skip] {key}")
                continue
            t0 = time.time()
            try:
                m = rate_point(a_ck, b_ck, scene, work)
            except Exception as ex:
                m = {"error": f"{type(ex).__name__}: {str(ex)[:150]}"}
            rec = {"key": key, "pair": name, "label": label, "dmax": dmax, **m,
                   "secs": round(time.time() - t0)}
            store.put(rec)
            report.line(f"{key:28} dmax {dmax:<7} hold {str(rec.get('hold_rate')):>6} "
                        f"reorient {str(rec.get('reorient_rate')):>6} "
                        f"cos|held {str(rec.get('cos_p50_held')):>7} {rec.get('error', '')}")
    runlib.Sentinel(ROOT / "logs/COMPLIANCE_RATE.DONE").write()
    print(f"[rate-sweep] COMPLETE -> {TXT_P}")


if __name__ == "__main__":
    main()
