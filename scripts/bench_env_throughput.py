#!/usr/bin/env python3
"""Where does the training wall clock actually go, and what moves it?

The standing conclusion (`project_rl_training_economics`) is ~10k steps/s, flat in
num_envs, GPU compute-saturated, so no knob helps. The first half is measured; the
explanation is not, and it is wrong. A live trainer iteration reports:

    Collection time: 7.908s     Learning time: 0.090s     Steps per second: 9218

so 98.9% of the wall clock is mjwarp stepping and the PPO update is free — no
learning-side knob (epochs, minibatches, lr, network width) can matter. Meanwhile
`nvidia-smi` shows 43-46% GPU utilisation with CUDA graph capture already ENABLED
(mjlab/sim/sim.py captures step/forward/reset/sense graphs). Kernel launch overhead is
therefore already collapsed, and a compute-saturated device would read ~100%.

What is left is that our model is tiny (15 joints, a handful of contacts) and MuJoCo's
solver is deep and sequential: `iterations x ls_iterations` dependent steps, each too
small to fill the device. That predicts something the standing conclusion denies —
adding envs should be nearly FREE until the kernels are wide enough to saturate.

This script measures it directly. It steps the real training env with zero actions and
reports steps/s, so the two candidate levers can be read off:

  --envs      does throughput scale with batch width? (the launch/occupancy question)
  --solver    what do the never-tuned solver iterations cost? (the depth question)

Solver changes alter contact dynamics and therefore the policy, so treat a win there as
a hypothesis to validate (re-run the open-loop anchor probe and confirm the reorient
still lands) rather than as a free speedup.

  uv run --extra rl --extra gpu python scripts/bench_env_throughput.py \
      --morphology-run results/phase1/landscape/m05_ik_cem --open-finger-from-keyframe
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import torch


def bench(cfg, num_envs, steps, warmup, label, sim_over=None):
    """steps/s for one config. Warmup covers kernel compile + graph capture, which
    otherwise lands entirely in the first timed iteration and reads as a slow config.

    `sim_over` patches the MuJoCo block AFTER `to_mjlab_cfg`, because the solver knobs
    live in `env_build`'s hardcoded SimulationCfg (iterations=10, ls_iterations=20,
    cone="elliptic") and are not exposed on MorphoHandEnvCfg."""
    from mjlab.envs import ManagerBasedRlEnv
    from morphohand.rl.env_cfg import to_mjlab_cfg

    cfg = dataclasses.replace(cfg, num_envs=num_envs)
    mj = to_mjlab_cfg(cfg)
    for k, v in (sim_over or {}).items():
        setattr(mj.sim.mujoco, k, v)
    env = ManagerBasedRlEnv(cfg=mj, device="cuda:0", render_mode=None)
    try:
        act = torch.zeros((num_envs, env.action_manager.total_action_dim), device="cuda:0")
        env.reset()
        for _ in range(warmup):
            env.step(act)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(act)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    finally:
        env.close()
        del env
        torch.cuda.empty_cache()
    sps = num_envs * steps / dt
    mem = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    print(f"  {label:28s} {sps:10,.0f} steps/s   ({dt:5.2f}s for {steps} iters, "
          f"peak {mem:4.1f} GiB)")
    return {"label": label, "num_envs": num_envs, "steps_per_s": sps,
            "wall_s": dt, "peak_gib": mem}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--morphology-run", type=Path, required=True)
    ap.add_argument("--open-finger-from-keyframe", action="store_true")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--envs", type=int, nargs="+", default=[1024, 3072, 6144, 12288])
    ap.add_argument("--solver", type=str, nargs="*", default=["10/20", "10/10", "5/10", "pyramidal"],
                    help="iterations/ls_iterations pairs (or 'pyramidal'), at --solver-envs. "
                         "Current shipped setting is 10/20 elliptic.")
    ap.add_argument("--solver-envs", type=int, default=3072)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    from morphohand.rl.deploy import make_env_cfg

    with (args.morphology_run / "summary.json").open() as f:
        keyframe = json.load(f).get("keyframe", "open_short_manual")
    bfc = tuple(float(v) for v in
                np.load(args.morphology_run / "best_rollout.npz")["best_finger_ctrl"].reshape(-1))
    base = make_env_cfg(args.morphology_run / "frozen_scene.xml", keyframe, args.morphology_run,
                        bfc, enable_target_axis=True, num_steps=240,
                        open_finger_from_keyframe=args.open_finger_from_keyframe, num_envs=1)
    from morphohand.rl.env_cfg import to_mjlab_cfg as _t
    _m = _t(dataclasses.replace(base, num_envs=1)).sim
    print(f"[bench] current sim: iterations={_m.mujoco.iterations} "
          f"ls_iterations={_m.mujoco.ls_iterations} cone={_m.mujoco.cone} "
          f"impratio={_m.mujoco.impratio} nconmax={_m.nconmax} njmax={_m.njmax} "
          f"timestep={_m.mujoco.timestep} decimation={base.decimation}")

    rows = []
    print("\n[bench] === BATCH WIDTH (does throughput scale with num_envs?) ===")
    for n in args.envs:
        try:
            rows.append(bench(base, n, args.steps, args.warmup, f"num_envs={n}"))
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"  num_envs={n:<22d} OOM/error: {str(e)[:60]}")
            torch.cuda.empty_cache()
            break

    if args.solver:
        print(f"\n[bench] === SOLVER DEPTH / CONE (at num_envs={args.solver_envs}) ===")
        print("        (these CHANGE CONTACT DYNAMICS — any win here is a hypothesis to")
        print("         validate against the open-loop anchor, not a free speedup)")
        for spec in args.solver:
            over = {}
            if spec == "pyramidal":
                over = {"cone": "pyramidal"}
            else:
                it, ls = (int(v) for v in spec.split("/"))
                over = {"iterations": it, "ls_iterations": ls}
            try:
                rows.append(bench(base, args.solver_envs, args.steps, args.warmup,
                                  f"solver {spec}", sim_over=over))
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"  solver {spec:<20s} error: {str(e)[:60]}")
                torch.cuda.empty_cache()

    if rows:
        best = max(rows, key=lambda r: r["steps_per_s"])
        base_row = next((r for r in rows if r["num_envs"] == 3072
                         and r["label"].startswith("num_envs")), rows[0])
        print(f"\n[bench] baseline (num_envs=3072): {base_row['steps_per_s']:,.0f} steps/s")
        print(f"[bench] best measured:            {best['steps_per_s']:,.0f} steps/s "
              f"({best['label']}) = {best['steps_per_s']/base_row['steps_per_s']:.2f}x")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=1))
        print(f"[bench] -> {args.output}")


if __name__ == "__main__":
    main()
