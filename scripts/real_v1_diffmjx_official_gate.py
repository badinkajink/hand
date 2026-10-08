#!/usr/bin/env python3
"""Probe the released DiffMJX fork on a real_v1 held screwdriver state.

Run this with the isolated environment documented in the result page. The
baseline, CFD, scan-loop and soft-collision switches belong to the authors'
fork; the CPU MuJoCo reference uses the same XML and initial state.
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
import real_v1_diffmjx_gate as gate  # noqa: E402


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
    scene, meta = gate.make_scene("D1", "bed", 100.0, no_post=True,
                                  tool_capsule=args.tool_capsule, tip=args.tip,
                                  contact_model=args.contact_model,
                                  pad_spacing_m=args.pad_spacing_mm * 0.001)
    model = mujoco.MjModel.from_xml_path(str(scene))
    supported_scene, _ = gate.make_scene("D1", "bed", 100.0,
                                         tip=args.tip, contact_model=args.contact_model,
                                         pad_spacing_m=args.pad_spacing_mm * 0.001)
    supported_model = mujoco.MjModel.from_xml_path(str(supported_scene))
    held = gate.clone_data(model, gate.grip_state(supported_model, meta, "D1", 1))
    obj = model.body(gate.rb.OBJ).id
    qadr = int(model.jnt_qposadr[model.body_jntadr[obj]])
    vadr = int(model.jnt_dofadr[model.body_jntadr[obj]])
    yaw_ids = np.array([model.actuator(f"a_{finger}_yaw").id for finger in gate.rb.FINGERS])
    cpu = gate.clone_data(model, held)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        mujoco.mj_step(model, cpu)
    cpu_rollout_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    mx = mjx.put_model(model)
    put_model_s = time.perf_counter() - t0
    print(f"put_model {put_model_s:.3f} s", flush=True)
    opt = mx.opt.replace(
        cfd_enable=args.cfd,
        cfd_solimp=jnp.array([0.0, 0.01, 0.05, 1.0, 2.0]),
        scan_loop=args.scan_loop,
        col_soft_enable=args.soft_collision,
        softjax_mode="smooth",
        col_softness=0.01,
    )
    mx = mx.replace(opt=opt)
    md = mjx.put_data(model, held)
    print("put_data complete", flush=True)
    if not args.float32:
        md = md.replace(contact=md.contact.replace(
            geom1=md.contact.geom1.astype(jnp.int64),
            geom2=md.contact.geom2.astype(jnp.int64),
            geom=md.contact.geom.astype(jnp.int64),
        ))

    @jax.jit
    def rollout(delta):
        state = md.replace(ctrl=md.ctrl.at[yaw_ids].add(delta))
        def step(carry, _):
            return mjx.step(mx, carry), None
        return jax.lax.scan(step, state, None, length=args.steps)[0]

    @jax.jit
    def metrics(delta):
        final = rollout(delta)
        quat = final.qpos[qadr + 3:qadr + 7]
        vertical_cos = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        return jnp.array([vertical_cos, final.qpos[qadr + 2],
                          final.qvel[vadr + 3], final.qfrc_constraint[vadr + 3]])

    zero = jnp.zeros(3)
    forward_fn = jax.jit(lambda state: mjx.forward(mx, state))
    t0 = time.perf_counter()
    initial = forward_fn(md)
    initial.qpos.block_until_ready()
    initial_s = time.perf_counter() - t0
    print(f"mjx.forward {initial_s:.3f} s", flush=True)
    t0 = time.perf_counter()
    final = rollout(zero)
    final.qpos.block_until_ready()
    first_s = time.perf_counter() - t0
    warm_times = []
    for _ in range(args.warm_repeats):
        t0 = time.perf_counter()
        rollout(zero).qpos.block_until_ready()
        warm_times.append(time.perf_counter() - t0)
    grad_fields = {}
    if not args.skip_gradient:
        eps = 0.01
        basis = np.eye(3)
        def cpu_metrics(delta):
            state = gate.clone_data(model, held)
            state.ctrl[yaw_ids] += delta
            for _ in range(args.steps):
                mujoco.mj_step(model, state)
            quat = state.qpos[qadr + 3:qadr + 7]
            return np.array([1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2),
                             state.qpos[qadr + 2], state.qvel[vadr + 3],
                             state.qfrc_constraint[vadr + 3]])
        if args.one_direction_gradient:
            vector = jnp.asarray(basis[args.direction_index])
            @jax.jit
            def directional(delta):
                return jax.jvp(metrics, (delta,), (vector,))[1]
            t0 = time.perf_counter()
            ad = directional(zero)
            ad.block_until_ready()
            gradient_compile_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            directional(zero).block_until_ready()
            gradient_warm_s = time.perf_counter() - t0
            fork_fd = (np.asarray(metrics(eps * vector)) -
                       np.asarray(metrics(-eps * vector))) / (2 * eps)
            t0 = time.perf_counter()
            cpu_fd = (cpu_metrics(eps * basis[args.direction_index]) -
                      cpu_metrics(-eps * basis[args.direction_index])) / (2 * eps)
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
            t0 = time.perf_counter()
            jacobian = jacobian_fn(zero)
            jacobian.block_until_ready()
            jacobian_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            jacobian_fn(zero).block_until_ready()
            jacobian_warm_s = time.perf_counter() - t0
            fd = np.column_stack([
                (np.asarray(metrics(jnp.asarray(eps * b))) -
                 np.asarray(metrics(jnp.asarray(-eps * b)))) / (2 * eps)
                for b in basis
            ])
            t0 = time.perf_counter()
            cpu_fd = np.column_stack([(cpu_metrics(eps * b) - cpu_metrics(-eps * b)) / (2 * eps)
                                      for b in basis])
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
    c = gate.contact_summary(model, held)
    contact = initial.contact
    tool_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] == obj}
    finger_gap = {finger: None for finger in gate.rb.FINGERS}
    for (g0, g1), distance in zip(np.asarray(contact.geom), np.asarray(contact.dist)):
        g0, g1 = int(g0), int(g1)
        if g0 not in tool_geoms and g1 not in tool_geoms:
            continue
        other = g1 if g0 in tool_geoms else g0
        name = model.body(model.geom_bodyid[other]).name
        for finger in gate.rb.FINGERS:
            if name.startswith(finger + "_"):
                prior = finger_gap[finger]
                finger_gap[finger] = float(distance) if prior is None else min(prior, float(distance))
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
        "cpu_initial_contact": c,
        "fork_initial_finger_dist_m": finger_gap,
        "fork_initial_contact_dist_m": np.asarray(contact.dist).tolist(),
        "fork_initial_contact_geom": np.asarray(contact.geom).tolist(),
        "cpu_tool_qpos": cpu.qpos[qadr:qadr + 7].tolist(),
        "fork_tool_qpos": np.asarray(final.qpos[qadr:qadr + 7]).tolist(),
        "max_qpos_abs_error": float(np.max(np.abs(cpu.qpos - np.asarray(final.qpos)))),
        "max_qvel_abs_error": float(np.max(np.abs(cpu.qvel - np.asarray(final.qvel)))),
        "fork_metrics": [float(1 - 2 * (np.asarray(final.qpos)[qadr + 4] ** 2 +
                                       np.asarray(final.qpos)[qadr + 5] ** 2)),
                         float(final.qpos[qadr + 2]), float(final.qvel[vadr + 3]),
                         float(final.qfrc_constraint[vadr + 3])],
        "metric_names": ["tool_vertical_cos", "tool_z_m", "tool_omega_x_rad_s",
                         "tool_contact_torque_x_Nm"],
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
