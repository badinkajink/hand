#!/usr/bin/env python3
"""How far can the shipped inline hand's a10 -> b33 handoff be pushed before it stops working?

    MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/sim2real_robustness_sweep.py

WHY. The inline arrangement is the one the prototype natively supports, so it is the one going
to hardware — and the only robustness axis ever measured on it is contact compliance, where it
is known to be fragile (reorient rate 0.94 at the trained stiffness, 0.06 one step harder).
Compliance is not the only thing a real hand gets wrong. The pad's friction is a material
guess, the tool's mass and diameter are nominal, and the object never lands where the reset
puts it. This walks each of those independently, with the SAME frozen policies, and reports
where the cliff is on each — so hardening effort goes where the cliff actually is instead of
where the last investigation happened to look.

PROTOCOL, and why it is not policy_eval_suite. b33 trains with a live-A reset on a 250-step
horizon, so a scripted-lift harness run long is out of distribution and returns 6% alignment
for a policy that works (this was measured, and discarded, when the two arrangements were
compared). The native protocol is the continuous A->B handoff in ONE env with no reset at the
seam, which is what `rate_point` below runs, batched, at the deploy condition (deterministic
mean actions -- the env-to-env spread is the GPU solver's own contact non-determinism).

Every point is n=N_ENVS rollouts, matching `compliance_rate_sweep.py` so its numbers drop into
the same table. Resumable: per-point records land in the JSON and finished keys are skipped.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

from morphohand.studies import runlib
from morphohand.studies.runlib import ROOT, RESULTS_RL as RL
from morphohand.studies.scene_mutate import Scene

MORPH = ROOT / "results/phase1/landscape/m05_ik_cem"
OUT = ROOT / "results/phase1/sim2real/robustness"
JSON_P = ROOT / "docs/experiments/SIM2REAL_ROBUSTNESS.json"
TXT_P = ROOT / "docs/experiments/SIM2REAL_ROBUSTNESS.txt"

A_CKPT = RL / "a10_20260702-1256-policyA_m05_ik10/tensorboard/model_609.pt"
B_CKPT = RL / "b33_20260702-1353-policyB_m05_reorient_ik10/tensorboard/model_270.pt"

N_ENVS = 32
TOTAL_STEPS = 300      # b33's own horizon is 250; 300 is the protocol the arrangements page used
HANDOFF = 40
TAIL = 50
HOLD_Z = 0.05          # post-handoff min height below which the tool is on its way to the floor
ALIGN_COS = 0.7


# --------------------------------------------------------------------------- the axes
# (axis, label, scene mutator or None, env-cfg overrides). One physical property moves per point
# and the baseline sits in every axis so each curve is readable on its own.
def axes() -> list[tuple[str, str, object, dict]]:
    pts: list[tuple[str, str, object, dict]] = [("baseline", "trained", None, {})]

    # Contact stiffness. dmax up = harder = a less compliant pad than the one b33 learned on.
    for dmin, dmax in ((0.98, 0.997), (0.983, 0.998), (0.985, 0.999)):
        pts.append(("compliance", f"dmax{dmax}", lambda s, a=dmin, b=dmax: s.set_solimp(a, b), {}))

    # Pad friction. The scene assumes 2.4 sliding on both tip and tool, which is an optimistic
    # silicone-on-steel number; the real pad could easily be half that.
    for f in (0.5, 0.7, 1.4):
        pts.append(("friction", f"x{f}", lambda s, k=f: s.scale_friction(k), {}))

    # Tool mass. The screwdriver is a 500 kg/m^3 stand-in; a real one is denser and top-heavy.
    for f in (0.6, 1.6, 2.5):
        pts.append(("mass", f"x{f}", lambda s, k=f: s.scale_object_density(k), {}))

    # Shaft diameter at fixed mass -- a grip-geometry change, not a load change.
    for f in (0.85, 1.15):
        pts.append(("shaft_radius", f"x{f}", lambda s, k=f: s.scale_object_radius(k), {}))

    # Where the tool lands. On hardware nothing places it to the millimetre.
    for x, y in ((0.002, 0.002), (0.005, 0.005)):
        pts.append(("spawn_xy", f"+-{int(x * 1000)}mm", None,
                    {"cube_spawn_x_jitter": x, "cube_spawn_y_jitter": y}))
    for yaw in (0.1, 0.3):
        pts.append(("spawn_yaw", f"+-{yaw}rad", None, {"cube_spawn_yaw_jitter": yaw}))
    return pts


def scene_dir(label: str, mutate) -> Path:
    """A morphology-run-shaped directory holding the mutated scene (deploy needs all three)."""
    d = OUT / label
    d.mkdir(parents=True, exist_ok=True)
    for f in ("best_rollout.npz", "summary.json"):
        (d / f).write_bytes((MORPH / f).read_bytes())
    sc = Scene(MORPH / "frozen_scene.xml")
    if mutate is not None:
        mutate(sc)
    sc.write(d / "frozen_scene.xml")
    return d


# --------------------------------------------------------------------------- one point
def rate_point(scene: Path, work: Path, cfg_over: dict) -> dict:
    """One batched A->B handoff. Fresh envs per call so nothing leaks between points."""
    import numpy as np
    import torch
    from morphohand.rl import deploy

    bfc = tuple(float(v) for v in
                np.load(scene / "best_rollout.npz")["best_finger_ctrl"].reshape(-1))
    keyframe = json.loads((scene / "summary.json").read_text()).get("keyframe",
                                                                   "open_short_manual")
    frozen = scene / "frozen_scene.xml"
    a_obs_dim = deploy.ckpt_obs_dim(A_CKPT)

    cfg = deploy.make_env_cfg(frozen, keyframe, scene, bfc, enable_target_axis=True,
                              num_steps=TOTAL_STEPS, open_finger_from_keyframe=True)
    cfg.num_envs = N_ENVS
    for k, v in cfg_over.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"env cfg has no field {k!r}")   # fail loud, not silently OOD
        setattr(cfg, k, v)

    env, wrapped, actor_b = deploy.build_actor(cfg, B_CKPT, work)
    cfg_a = deploy.make_env_cfg(frozen, keyframe, scene, bfc,
                                enable_target_axis=(a_obs_dim == 66), num_steps=10,
                                open_finger_from_keyframe=True)
    env_a, _, actor_a = deploy.build_actor(cfg_a, A_CKPT, work)
    env_a.close()

    obs_td, _ = wrapped.reset()
    obj = env.scene["cube"]
    minz_post = torch.full((N_ENVS,), 1e9, device=env.device)
    peak_cos = torch.full((N_ENVS,), -1.0, device=env.device)
    t_align = torch.full((N_ENVS,), float("nan"), device=env.device)
    cos_hist = []
    with torch.no_grad():
        for step in range(TOTAL_STEPS):
            a = (deploy.act(actor_a, obs_td["actor"][:, :a_obs_dim]) if step < HANDOFF
                 else deploy.act_b(actor_b, obs_td, stochastic=False))
            obs_td, *_ = wrapped.step(a)
            q = obj.data.root_link_quat_w                      # (N,4) wxyz
            cos = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)    # tool axis vs vertical
            z = obj.data.root_link_pos_w[:, 2]
            if step >= HANDOFF:
                minz_post = torch.minimum(minz_post, z)
                peak_cos = torch.maximum(peak_cos, cos)
                newly = torch.isnan(t_align) & (cos > ALIGN_COS)
                t_align = torch.where(newly, torch.full_like(t_align, float(step)), t_align)
            cos_hist.append(cos.clone())

    tail_cos = torch.stack(cos_hist[-TAIL:]).mean(dim=0)
    held = minz_post > HOLD_Z
    reor = held & (tail_cos > ALIGN_COS)
    aligned = ~torch.isnan(t_align)
    res = dict(
        n=N_ENVS,
        hold_rate=round(float(held.float().mean()), 3),
        reorient_rate=round(float(reor.float().mean()), 3),
        align_rate=round(float(aligned.float().mean()), 3),
        peak_cos_mean=round(float(peak_cos.mean()), 3),
        cos_p50_held=(round(float(tail_cos[held].median()), 3) if held.any() else None),
        t_align_mean=(round(float(t_align[aligned].mean()), 1) if aligned.any() else None),
        minz_p50=round(float(minz_post.median()), 4),
    )
    env.close()
    del env, wrapped, actor_a, actor_b
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main() -> None:
    store = runlib.RecordStore(JSON_P, key_field="key")
    report = runlib.TxtReport(
        TXT_P, f"# sim2real robustness sweep {time.strftime('%Y-%m-%d %H:%M')}  "
               f"a10->b33 (inline arrangement), n={N_ENVS}/point, {TOTAL_STEPS} steps\n"
               f"# one physical axis moved per point; deterministic actions, solver-noise spread\n")
    work = ROOT / "logs" / "_robustness_tmp"
    work.mkdir(parents=True, exist_ok=True)
    for axis, label, mutate, cfg_over in axes():
        key = f"{axis}:{label}"
        if key in store:
            print(f"[skip] {key}")
            continue
        t0 = time.time()
        try:
            m = rate_point(scene_dir(f"{axis}_{label}".replace("/", "_"), mutate), work, cfg_over)
        except Exception as ex:                      # a broken point must not kill the sweep
            m = {"error": f"{type(ex).__name__}: {str(ex)[:160]}"}
        rec = {"key": key, "axis": axis, "label": label, **m, "secs": round(time.time() - t0)}
        store.put(rec)
        report.line(f"{key:26} hold {str(rec.get('hold_rate')):>6} "
                    f"reorient {str(rec.get('reorient_rate')):>6} "
                    f"peak {str(rec.get('peak_cos_mean')):>6} "
                    f"cos|held {str(rec.get('cos_p50_held')):>7} "
                    f"minz {str(rec.get('minz_p50')):>7} {rec.get('error', '')}")
        print(f"[{key}] {rec}")
    runlib.Sentinel(ROOT / "logs/SIM2REAL_ROBUSTNESS.DONE").write()
    print(f"[robustness] COMPLETE -> {TXT_P}")


if __name__ == "__main__":
    main()
