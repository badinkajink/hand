#!/usr/bin/env python3
"""DiffMJX directional contact derivatives on the packed real_v1 D1 grip.

Run in the isolated DiffMJX environment. The scene and held state come from real_v1_d1_continuation.py (packed pads
that touch only the tool); the four outputs are its METRIC_NAMES. Without --fork-fd-only the script compiles one JVP
with respect to the index yaw target (with --all-directions, all three yaw targets) and compares it with CPU central
differences; --fork-fd-only replaces the JVP by fork central differences at the steps of --fd-eps-list.
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
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--pad-spacing-mm", type=float, default=1.0)
    parser.add_argument("--cfd", action="store_true")
    parser.add_argument("--scan-loop", action="store_true")
    parser.add_argument("--fork-fd-only", action="store_true")
    parser.add_argument("--all-directions", action="store_true",
                        help="Differentiate all three finger-yaw targets")
    parser.add_argument("--fd-eps-list", default="0.01",
                        help="Comma-separated central-difference steps in radians")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    spacing = args.pad_spacing_mm * 0.001
    scene, meta, model, held = d1.held_scene("tpu6", "padsT", 1, args.pad_spacing_mm)
    qadr, vadr = d1.tool_addresses(model)
    yaw_ids = np.array([model.actuator(f"a_{f}_yaw").id for f in d1.FINGERS])
    mx, md, put_model_s = d1.mjx_setup(model, held, cfd=args.cfd, scan_loop=args.scan_loop)
    steps = d1.mjx_rollout(mx, args.steps)
    basis = jnp.array([0.0, 1.0, 0.0])

    def metrics(delta):
        final = steps(md.replace(ctrl=md.ctrl.at[yaw_ids].add(delta)))
        return jnp.array(d1.metric_values(final.qpos, final.qvel, final.qfrc_constraint, qadr, vadr))

    eps_list = [float(value) for value in args.fd_eps_list.split(",")]
    if not eps_list or any(value <= 0 for value in eps_list):
        parser.error("finite-difference steps must be positive")
    eps = eps_list[0]
    zero = jnp.zeros(3)
    fork_fd_sweep = None
    ad_jacobian = None
    if args.fork_fd_only:
        forward = jax.jit(metrics)
        t0 = time.perf_counter()
        base = forward(zero)
        fork_fd_sweep = {}
        for step_size in eps_list:
            plus = forward(step_size * basis)
            minus = forward(-step_size * basis)
            minus.block_until_ready()
            fork_fd_sweep[f"{step_size:g}"] = ((np.asarray(plus) - np.asarray(minus)) / (2 * step_size)).tolist()
        first_s = time.perf_counter() - t0
        fork_fd = np.asarray(fork_fd_sweep[f"{eps:g}"])
        print(f"forward and finite difference compile {first_s:.3f} s", flush=True)
        warm_s = d1.timed(forward, zero)[1]
        ad_np = None
    else:
        directional = jax.jit(lambda delta, tangent: jax.jvp(metrics, (delta,), (tangent,)))
        (base, ad), first_s = d1.timed(directional, zero, basis)
        print(f"directional compile {first_s:.3f} s", flush=True)
        t0 = time.perf_counter()
        if args.all_directions:
            ad_jacobian = np.column_stack([np.asarray(directional(zero, direction)[1]) for direction in jnp.eye(3)])
        else:
            directional(zero, basis)[1].block_until_ready()
        warm_s = time.perf_counter() - t0
        ad_np = np.asarray(ad)
        fork_fd = None

    def cpu_metrics(delta):
        return d1.cpu_metrics(model, held, args.steps, yaw_ids, delta)

    vector = np.array([0.0, 1.0, 0.0])
    t0 = time.perf_counter()
    cpu_base = cpu_metrics(np.zeros(3))
    cpu_fd_sweep = {f"{step_size:g}": d1.central_difference(cpu_metrics, np.zeros(3), [vector], step_size)[:, 0].tolist()
                    for step_size in eps_list}
    cpu_fd_s = time.perf_counter() - t0      # the base and the directional sweep only (corrected 2026-10-11)
    cpu_fd_evals = 1 + 2 * len(eps_list)
    cpu_fd = np.asarray(cpu_fd_sweep[f"{eps:g}"])
    cpu_fd_jacobian = None
    cpu_jacobian_s = None
    local_search = None
    if args.all_directions:
        t0 = time.perf_counter()
        cpu_fd_jacobian = d1.central_difference(cpu_metrics, np.zeros(3), np.eye(3), eps)
        cpu_jacobian_s = time.perf_counter() - t0
        radius = 0.1
        ad_direction = radius * ad_jacobian[0] / np.linalg.norm(ad_jacobian[0])
        cpu_direction = radius * cpu_fd_jacobian[0] / np.linalg.norm(cpu_fd_jacobian[0])
        local_search = {
            "radius_rad": radius,
            "ad_direction_rad": ad_direction.tolist(),
            "cpu_fd_direction_rad": cpu_direction.tolist(),
            "direction_cosine": float(np.dot(ad_direction, cpu_direction) / radius**2),
            "cpu_score_zero": float(cpu_base[0]),
            "cpu_score_ad_direction": float(cpu_metrics(ad_direction)[0]),
            "cpu_score_fd_direction": float(cpu_metrics(cpu_direction)[0]),
        }
    out = {
        "scene": str(scene.relative_to(ROOT)),
        "base_scene": meta["base_scene"],
        "mujoco_version": mujoco.__version__, "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "mjx_package": str(Path(mjx.__file__).resolve()),
        "n_pads": meta["n_pads"], "pad_spacing_m": spacing,
        "steps": args.steps, "timestep_s": float(model.opt.timestep),
        "cfd_enable": args.cfd, "scan_loop": args.scan_loop,
        "mode": "fork_fd_only" if args.fork_fd_only else "directional_jvp",
        "direction_index": 1, "finite_difference_step_rad": eps,
        "metric_names": list(d1.METRIC_NAMES),
        "fork_base_metrics": np.asarray(base).tolist(),
        "cpu_base_metrics": cpu_base.tolist(),
        "ad_nonfinite_count": None if ad_np is None else int(np.size(ad_np) - np.isfinite(ad_np).sum()),
        "ad_directional_derivative": ad_np.tolist() if ad_np is not None and np.isfinite(ad_np).all() else None,
        "ad_jacobian": None if ad_jacobian is None else ad_jacobian.tolist(),
        "cpu_fd_jacobian": None if cpu_fd_jacobian is None else cpu_fd_jacobian.tolist(),
        "local_search": local_search,
        "fork_fd_directional_derivative": None if fork_fd is None else fork_fd.tolist(),
        "cpu_fd_directional_derivative": cpu_fd.tolist(),
        "fork_fd_sweep": fork_fd_sweep,
        "cpu_fd_sweep": cpu_fd_sweep,
        "put_model_s": put_model_s,
        "gradient_compile_s": first_s, "gradient_warm_s": warm_s,
        # until 2026-10-11 cpu_fd_3_evals_s also timed the Jacobian and the local search (12 evaluations)
        "cpu_fd_evals": cpu_fd_evals, "cpu_fd_s": cpu_fd_s,
        "cpu_fd_3_evals_s": cpu_fd_s if cpu_fd_evals == 3 else None,
        "cpu_jacobian_6_evals_s": cpu_jacobian_s,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
