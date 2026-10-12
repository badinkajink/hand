#!/usr/bin/env python3
"""Probe the released DiffMJX fork on a real_v1 held screwdriver state.

Run this with the isolated environment of docs/experiments/20261007-diffmjx/README.md. The CFD, scan-loop and
smooth-collision switches belong to the authors' fork (real_v1_d1_continuation.mjx_setup sets them); the CPU MuJoCo
reference uses the same XML and initial state. The four outputs are real_v1_d1_continuation.METRIC_NAMES, the last two
about the tool's body x axis.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--cfd", action="store_true")
    parser.add_argument("--scan-loop", action="store_true")
    parser.add_argument("--soft-collision", action="store_true")
    parser.add_argument("--tool-capsule", action="store_true")
    parser.add_argument("--tip", choices=("sphere", "tpu6"), default="sphere")
    parser.add_argument("--contact-model", choices=("pt", "pads", "padsT"), default="pt")
    parser.add_argument("--pad-spacing-mm", type=float, default=1.0)
    parser.add_argument("--skip-gradient", action="store_true")
    parser.add_argument("--one-direction-gradient", action="store_true")
    parser.add_argument("--direction-index", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--float32", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import jax
    jax.config.update("jax_enable_x64", not args.float32)
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    if args.contact_model in ("pads", "padsT") and args.tip != "tpu6":
        parser.error("sphere pads require --tip tpu6")
    scene, meta, model, held = d1.held_scene(args.tip, args.contact_model, 1, args.pad_spacing_mm,
                                             tool_capsule=args.tool_capsule)
    qadr, vadr = d1.tool_addresses(model)
    yaw_ids = np.array([model.actuator(f"a_{finger}_yaw").id for finger in d1.FINGERS])
    cpu = d1.clone_data(model, held)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        mujoco.mj_step(model, cpu)
    cpu_rollout_s = time.perf_counter() - t0

    mx, md, put_model_s = d1.mjx_setup(model, held, cfd=args.cfd, scan_loop=args.scan_loop,
                                       soft_collision=args.soft_collision)
    print(f"put_model {put_model_s:.3f} s", flush=True)
    steps = d1.mjx_rollout(mx, args.steps)

    @jax.jit
    def rollout(delta):
        return steps(md.replace(ctrl=md.ctrl.at[yaw_ids].add(delta)))

    @jax.jit
    def metrics(delta):
        final = rollout(delta)
        return jnp.array(d1.metric_values(final.qpos, final.qvel, final.qfrc_constraint, qadr, vadr))

    def cpu_metrics(delta):
        return d1.cpu_metrics(model, held, args.steps, yaw_ids, delta)

    zero = jnp.zeros(3)
    initial, initial_s = d1.timed(jax.jit(lambda state: mjx.forward(mx, state)), md)
    print(f"mjx.forward {initial_s:.3f} s", flush=True)
    final, first_s = d1.timed(rollout, zero)
    warm_times = [d1.timed(rollout, zero)[1] for _ in range(args.warm_repeats)]
    grad_fields = {}
    if not args.skip_gradient:
        eps = 0.01
        basis = np.eye(3)
        if args.one_direction_gradient:
            vector = jnp.asarray(basis[args.direction_index])
            directional = jax.jit(lambda delta: jax.jvp(metrics, (delta,), (vector,))[1])
            ad, gradient_compile_s = d1.timed(directional, zero)
            gradient_warm_s = d1.timed(directional, zero)[1]
            fork_fd = d1.central_difference(lambda u: np.asarray(metrics(jnp.asarray(u))), np.zeros(3),
                                            basis[[args.direction_index]], eps)[:, 0]
            t0 = time.perf_counter()
            cpu_fd = d1.central_difference(cpu_metrics, np.zeros(3), basis[[args.direction_index]], eps)[:, 0]
            cpu_fd_s = time.perf_counter() - t0
            ad_np = np.asarray(ad)
            grad_fields = {
                "direction_index": args.direction_index,
                "ad_nonfinite_count": int(np.size(ad_np) - np.isfinite(ad_np).sum()),
                "ad_directional_derivative": ad_np.tolist() if np.isfinite(ad_np).all() else None,
                "fork_fd_directional_derivative": fork_fd.tolist(),
                "cpu_fd_directional_derivative": cpu_fd.tolist(),
                "cpu_fd_2_evals_s": cpu_fd_s,
                "gradient_compile_s": gradient_compile_s,
                "gradient_warm_s": gradient_warm_s,
            }
        else:
            jacobian_fn = jax.jit(jax.jacfwd(metrics))
            jacobian, jacobian_s = d1.timed(jacobian_fn, zero)
            jacobian_warm_s = d1.timed(jacobian_fn, zero)[1]
            fd = d1.central_difference(lambda u: np.asarray(metrics(jnp.asarray(u))), np.zeros(3), basis, eps)
            t0 = time.perf_counter()
            cpu_fd = d1.central_difference(cpu_metrics, np.zeros(3), basis, eps)
            cpu_fd_s = time.perf_counter() - t0
            j = np.asarray(jacobian)
            grad_fields = {
                "ad_nonfinite_count": int(np.size(j) - np.isfinite(j).sum()),
                "ad_jacobian": j.tolist() if np.isfinite(j).all() else None,
                "fork_fd_jacobian": fd.tolist(),
                "cpu_fd_jacobian": cpu_fd.tolist(),
                "cpu_fd_6_evals_s": cpu_fd_s,
                "jacobian_compile_s": jacobian_s,
                "jacobian_warm_s": jacobian_warm_s,
            }
    contact, _ = d1.mjx_contact(initial)
    fq = np.asarray(final.qpos)
    result = {
        "scene": str(scene.relative_to(ROOT)),
        "base_scene": meta["base_scene"],
        "tool_geom": "capsule" if args.tool_capsule else "cylinder",
        "tip": args.tip, "contact_model": args.contact_model,
        "n_pads": meta.get("n_pads"),
        "pad_spacing_m": meta.get("pad_spacing_m"),
        "steps": args.steps,
        "timestep_s": float(model.opt.timestep),
        "mujoco_version": mujoco.__version__,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "mjx_package": str(Path(mjx.__file__).resolve()),
        "cfd_enable": args.cfd,
        "scan_loop": args.scan_loop,
        "col_soft_enable": args.soft_collision,
        "cpu_initial_contact": d1.contact_summary(model, held),
        "fork_initial_finger_dist_m": d1.mjx_finger_gaps(model, initial),
        "fork_initial_contact_dist_m": np.asarray(contact.dist).tolist(),
        "fork_initial_contact_geom": np.asarray(contact.geom).tolist(),
        "cpu_tool_qpos": cpu.qpos[qadr:qadr + 7].tolist(),
        "fork_tool_qpos": fq[qadr:qadr + 7].tolist(),
        "max_qpos_abs_error": float(np.max(np.abs(cpu.qpos - fq))),
        "max_qvel_abs_error": float(np.max(np.abs(cpu.qvel - np.asarray(final.qvel)))),
        "fork_metrics": [float(v) for v in d1.metric_values(fq, np.asarray(final.qvel),
                                                            np.asarray(final.qfrc_constraint), qadr, vadr)],
        "metric_names": list(d1.METRIC_NAMES),
        "initial_compile_s": initial_s,
        "put_model_s": put_model_s,
        "rollout_compile_s": first_s,
        "rollout_warm_s": warm_times,
        "rollout_warm_median_s": float(np.median(warm_times)) if warm_times else None,
        "cpu_rollout_s": cpu_rollout_s,
        **grad_fields,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ("fork_initial_contact_dist_m", "fork_initial_contact_geom")},
                     indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
