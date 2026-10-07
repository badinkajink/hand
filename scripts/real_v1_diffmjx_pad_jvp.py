#!/usr/bin/env python3
"""One DiffMJX directional contact derivative on the packed real_v1 D1 grip."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import real_v1_diffmjx_gate as gate  # noqa: E402


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
    scene, meta = gate.make_scene("D1", "bed", 100.0, no_post=True, tip="tpu6",
                                  contact_model="padsT", pad_spacing_m=spacing)
    model = mujoco.MjModel.from_xml_path(str(scene))
    supported, _ = gate.make_scene("D1", "bed", 100.0, tip="tpu6",
                                    contact_model="padsT", pad_spacing_m=spacing)
    held = gate.clone_data(model, gate.grip_state(mujoco.MjModel.from_xml_path(str(supported)),
                                                  meta, "D1", 1))
    obj = model.body(gate.rb.OBJ).id
    qadr = int(model.jnt_qposadr[model.body_jntadr[obj]])
    vadr = int(model.jnt_dofadr[model.body_jntadr[obj]])
    yaw_ids = np.array([model.actuator(f"a_{f}_yaw").id for f in gate.rb.FINGERS])
    t0 = time.perf_counter()
    mx = mjx.put_model(model)
    put_model_s = time.perf_counter() - t0
    mx = mx.replace(opt=mx.opt.replace(
        cfd_enable=args.cfd, cfd_solimp=jnp.array([0.0, 0.01, 0.05, 1.0, 2.0]),
        scan_loop=args.scan_loop, col_soft_enable=False, softjax_mode="smooth",
        col_softness=0.01))
    md = mjx.put_data(model, held)
    md = md.replace(contact=md.contact.replace(
        geom1=md.contact.geom1.astype(jnp.int64),
        geom2=md.contact.geom2.astype(jnp.int64),
        geom=md.contact.geom.astype(jnp.int64)))
    basis = jnp.array([0.0, 1.0, 0.0])

    def metrics(delta):
        state = md.replace(ctrl=md.ctrl.at[yaw_ids].add(delta))
        def step(carry, _):
            return mjx.step(mx, carry), None
        final = jax.lax.scan(step, state, None, length=args.steps)[0]
        quat = final.qpos[qadr + 3:qadr + 7]
        return jnp.array([1 - 2 * (quat[1] ** 2 + quat[2] ** 2),
                          final.qpos[qadr + 2], final.qvel[vadr + 3],
                          final.qfrc_constraint[vadr + 3]])

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
            fork_fd_sweep[f"{step_size:g}"] = ((np.asarray(plus) - np.asarray(minus)) /
                                                 (2 * step_size)).tolist()
        first_s = time.perf_counter() - t0
        fork_fd = np.asarray(fork_fd_sweep[f"{eps:g}"])
        print(f"forward and finite difference compile {first_s:.3f} s", flush=True)
        t0 = time.perf_counter()
        forward(zero).block_until_ready()
        warm_s = time.perf_counter() - t0
        ad_np = None
    else:
        @jax.jit
        def directional(delta, tangent):
            return jax.jvp(metrics, (delta,), (tangent,))
        t0 = time.perf_counter()
        base, ad = directional(zero, basis)
        ad.block_until_ready()
        first_s = time.perf_counter() - t0
        print(f"directional compile {first_s:.3f} s", flush=True)
        t0 = time.perf_counter()
        if args.all_directions:
            directions = jnp.eye(3)
            columns = []
            for direction in directions:
                _, column = directional(zero, direction)
                columns.append(np.asarray(column))
            ad_jacobian = np.column_stack(columns)
        else:
            directional(zero, basis)[1].block_until_ready()
        warm_s = time.perf_counter() - t0
        ad_np = np.asarray(ad)
        fork_fd = None

    def cpu_metrics(delta):
        state = gate.clone_data(model, held)
        state.ctrl[yaw_ids] += delta
        for _ in range(args.steps):
            mujoco.mj_step(model, state)
        quat = state.qpos[qadr + 3:qadr + 7]
        return np.array([1 - 2 * (quat[1] ** 2 + quat[2] ** 2),
                         state.qpos[qadr + 2], state.qvel[vadr + 3],
                         state.qfrc_constraint[vadr + 3]])

    vector = np.array([0.0, 1.0, 0.0])
    t0 = time.perf_counter()
    cpu_base = cpu_metrics(np.zeros(3))
    cpu_fd_sweep = {f"{step_size:g}": ((cpu_metrics(step_size * vector) -
                                       cpu_metrics(-step_size * vector)) /
                                      (2 * step_size)).tolist() for step_size in eps_list}
    cpu_fd = np.asarray(cpu_fd_sweep[f"{eps:g}"])
    cpu_fd_jacobian = None
    local_search = None
    if args.all_directions:
        cpu_fd_jacobian = np.column_stack([
            (cpu_metrics(eps * direction) - cpu_metrics(-eps * direction)) / (2 * eps)
            for direction in np.eye(3)
        ])
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
    cpu_fd_s = time.perf_counter() - t0
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
        "metric_names": ["tool_vertical_cos", "tool_z_m", "tool_omega_x_rad_s",
                         "tool_contact_torque_x_Nm"],
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
        "cpu_fd_3_evals_s": cpu_fd_s,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
