#!/usr/bin/env python3
"""Measure a local DiffMJX control derivative immediately before index recontact.

Replays a 400 ms plan (by default the 1 mm packed-pad CEM release plan of 2026-10-07) on 2 mm packed pads in CPU
MuJoCo to --start-ms, transfers that state to the released fork, and differentiates the next --steps steps with
respect to the index MCP target; fork and CPU central differences check the derivative. Scene, plan schedule and
outputs come from real_v1_d1_continuation.py. Run in the isolated DiffMJX environment.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import real_v1_d1_continuation as d1  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--method", default="cem")
    parser.add_argument("--start-ms", type=int, default=280)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfd", action="store_true")
    parser.add_argument("--scan-loop", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from mujoco import mjx

    x = np.asarray(json.loads(args.plans.read_text())["candidates"][args.method]["x"])
    scene = d1.Scene("tpu6", "padsT", 1, pad_spacing_mm=2)
    model = scene.model
    state = scene.state_at(x, args.start_ms)
    index_id = model.actuator("a_index_mcp").id
    qa, vadr = scene.qadr, scene.vadr
    initial_cpu = d1.contact_summary(model, state)

    def cpu_metrics(delta):
        d = d1.clone_data(model, state)
        d.ctrl[index_id] += delta
        for _ in range(args.steps):
            scene.mujoco.mj_step(model, d)
        return np.array(d1.metric_values(d.qpos, d.qvel, d.qfrc_constraint, qa, vadr)), d1.contact_summary(model, d)

    cpu_zero, cpu_contact = cpu_metrics(0.0)
    cpu_fd = {}
    t0 = time.perf_counter()
    for eps in (0.01, 0.003, 0.001):
        plus, cp = cpu_metrics(eps)
        minus, cm = cpu_metrics(-eps)
        cpu_fd[str(eps)] = {"derivative": ((plus - minus) / (2 * eps)).tolist(),
                            "plus_index_force_N": cp["finger_normal_force_N"]["index"],
                            "minus_index_force_N": cm["finger_normal_force_N"]["index"]}
    cpu_fd_s = (time.perf_counter() - t0) / 3          # one central difference with its contact summaries
    print(f"CPU index force {initial_cpu['finger_normal_force_N']['index']:.3f} -> "
          f"{cpu_contact['finger_normal_force_N']['index']:.3f} N at {args.start_ms + args.steps} ms", flush=True)

    mx, md, put_model_s = d1.mjx_setup(model, state, cfd=args.cfd, scan_loop=args.scan_loop, fork_extras=False)
    jinitial, forward_compile_s = d1.timed(jax.jit(lambda d: mjx.forward(mx, d)), md)
    steps = d1.mjx_rollout(mx, args.steps)

    @jax.jit
    def metrics(delta):
        final = steps(md.replace(ctrl=md.ctrl.at[index_id].add(delta)))
        return jnp.array(d1.metric_values(final.qpos, final.qvel, final.qfrc_constraint, qa, vadr))

    derivative = jax.jit(lambda delta: jax.jvp(metrics, (delta,), (jnp.array(1.0),))[1])
    zero = jnp.array(0.0)
    base, rollout_compile_s = d1.timed(metrics, zero)
    ad, derivative_compile_s = d1.timed(derivative, zero)
    derivative_warm_s = d1.timed(derivative, zero)[1]
    eps = .01
    fork_fd = (np.asarray(metrics(jnp.array(eps))) - np.asarray(metrics(jnp.array(-eps)))) / (2 * eps)
    out = {"backend": "released DiffMJX fork of MJX 3.3.1", "jax_backend": jax.default_backend(),
           "precision": "float64", "scene": str(scene.scene), "plans": str(args.plans),
           "method": args.method, "pad_spacing_mm": 2, "start_ms": args.start_ms,
           "steps": args.steps, "control": "index MCP target bias (rad)", "cfd": args.cfd,
           "scan_loop": args.scan_loop, "metric_names": list(d1.METRIC_NAMES),
           "initial_cpu_index_force_N": initial_cpu["finger_normal_force_N"]["index"],
           "initial_fork_index_min_gap_m": d1.mjx_finger_gaps(model, jinitial)["index"],
           "final_cpu_index_force_N": cpu_contact["finger_normal_force_N"]["index"],
           "cpu_zero": cpu_zero.tolist(), "fork_zero": np.asarray(base).tolist(),
           "cpu_fd": cpu_fd, "cpu_fd_s": cpu_fd_s, "fork_fd_0.01": fork_fd.tolist(),
           "fork_ad": np.asarray(ad).tolist(), "put_model_s": put_model_s,
           "forward_compile_s": forward_compile_s, "rollout_compile_s": rollout_compile_s,
           "derivative_compile_s": derivative_compile_s,
           "derivative_warm_s": derivative_warm_s}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
