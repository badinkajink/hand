#!/usr/bin/env python3
"""Check MJX contact dynamics at a replayed real_v1 grasp before optimizer studies.

The scene, the held state and the outputs come from real_v1_d1_continuation.py: the deployed D1 plan's grip, the palm
display plate made non-colliding because MJX 3.6 does not implement cylinder-box collisions (the plate has no
contacts in this grasp), and four tool quantities, the last two about the tool's body x axis. This script runs the
installed MJX (native 3.6 in the project environment) against CPU MuJoCo from the same state, and with --gradient its
forward-mode Jacobian with respect to the three yaw targets against CPU central differences.
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
    parser.add_argument("--hand", default="D1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--numerics", choices=("bed", "scene"), default="bed")
    parser.add_argument("--impratio", type=float, default=100.0)
    parser.add_argument("--no-post", action="store_true")
    parser.add_argument("--tool-capsule", action="store_true")
    parser.add_argument("--tip", choices=("sphere", "tpu6"), default="sphere")
    parser.add_argument("--contact-model", choices=("pt", "pads", "padsT"), default="pt")
    parser.add_argument("--pad-spacing-mm", type=float, default=1.0)
    parser.add_argument("--gradient", action="store_true")
    parser.add_argument("--warm-repeats", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/experiments/20261007-diffmjx/20261007-mjx_gate.json")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    if args.contact_model in ("pads", "padsT") and args.tip != "tpu6":
        parser.error("sphere pads require --tip tpu6")
    spacing = args.pad_spacing_mm * 0.001
    scene, meta = d1.make_scene(args.hand, args.numerics, args.impratio, args.no_post, args.tool_capsule,
                                args.tip, args.contact_model, spacing)
    model = mujoco.MjModel.from_xml_path(str(scene))
    if args.no_post or args.tool_capsule:
        # the grip runs on the post with the cylinder tool; the state is then cloned into the adapted scene
        supported, _ = d1.make_scene(args.hand, args.numerics, args.impratio, tip=args.tip,
                                     contact_model=args.contact_model, pad_spacing_m=spacing)
        held = d1.clone_data(model, d1.grip_state(mujoco.MjModel.from_xml_path(str(supported)), meta, args.hand,
                                                  args.seed))
    else:
        held = d1.grip_state(model, meta, args.hand, args.seed)
    qadr, vadr = d1.tool_addresses(model)
    held_contacts = d1.contact_summary(model, held)
    assert held_contacts["palm_plate_contacts"] == 0
    print("held state", held_contacts, flush=True)

    cpu = d1.clone_data(model, held)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        mujoco.mj_step(model, cpu)
    cpu_s = time.perf_counter() - t0

    mx, md, put_model_s = d1.mjx_setup(model, held)
    print(f"put_model {put_model_s:.3f} s", flush=True)
    forward = jax.jit(lambda data: mjx.forward(mx, data))
    jinitial, forward_compile_s = d1.timed(forward, md)
    print("mjx.forward compiled", flush=True)
    rollout = jax.jit(d1.mjx_rollout(mx, args.steps))
    jdata, mjx_first_s = d1.timed(rollout, md)
    warm_times = [d1.timed(rollout, md)[1] for _ in range(args.warm_repeats)]

    jpos, jvel = np.asarray(jdata.qpos), np.asarray(jdata.qvel)
    result = {
        "scene": str(scene.relative_to(ROOT)), "base_scene": meta["base_scene"],
        "hand": args.hand, "seed": args.seed, "steps": args.steps,
        "numerics": args.numerics, "impratio": args.impratio,
        "post_collides": not args.no_post,
        "tool_geom": "capsule" if args.tool_capsule else "cylinder",
        "tip": args.tip, "contact_model": args.contact_model,
        "n_pads": meta.get("n_pads"),
        "pad_spacing_m": meta.get("pad_spacing_m"),
        "mujoco_version": mujoco.__version__, "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(), "nq": model.nq, "nv": model.nv,
        "ngeom": model.ngeom, "timestep_s": model.opt.timestep,
        "put_model_s": put_model_s,
        "initial_contact": held_contacts,
        "mjx_initial_contact": d1.mjx_finger_forces(model, jinitial),
        "cpu_contact": d1.contact_summary(model, cpu),
        "mjx_contact": d1.mjx_finger_forces(model, jdata),
        "cpu_tool_qpos": cpu.qpos[qadr:qadr + 7].tolist(),
        "mjx_tool_qpos": jpos[qadr:qadr + 7].tolist(),
        "cpu_tool_qvel": cpu.qvel[vadr:vadr + 6].tolist(),
        "mjx_tool_qvel": jvel[vadr:vadr + 6].tolist(),
        "max_qpos_abs_error": float(np.max(np.abs(cpu.qpos - jpos))),
        "max_qvel_abs_error": float(np.max(np.abs(cpu.qvel - jvel))),
        "cpu_rollout_s": cpu_s, "mjx_first_s": mjx_first_s,
        "mjx_warm_s": float(np.median(warm_times)),
        "mjx_warm_repeats_s": warm_times, "forward_compile_s": forward_compile_s,
    }

    if args.gradient:
        yaw_ids = np.array([model.actuator(f"a_{finger}_yaw").id for finger in d1.FINGERS])

        @jax.jit
        def mjx_metrics(delta):
            final = rollout(md.replace(ctrl=md.ctrl.at[yaw_ids].add(delta)))
            return jnp.array(d1.metric_values(final.qpos, final.qvel, final.qfrc_constraint, qadr, vadr))

        def cpu_metrics(delta):
            return d1.cpu_metrics(model, held, args.steps, yaw_ids, delta)

        zero = jnp.zeros(3)
        grad_fn = jax.jit(jax.jacfwd(mjx_metrics))
        jac, jac_first_s = d1.timed(grad_fn, zero)
        jac, jac_warm_s = d1.timed(grad_fn, zero)
        jac_np = np.asarray(jac)
        eps = 0.01
        basis = np.eye(3)
        t0 = time.perf_counter()
        cpu_base = cpu_metrics(np.zeros(3))
        cpu_fd = d1.central_difference(cpu_metrics, np.zeros(3), basis, eps)
        cpu_fd_s = time.perf_counter() - t0
        mjx_base = np.asarray(mjx_metrics(zero))
        mjx_fd = d1.central_difference(lambda u: np.asarray(mjx_metrics(jnp.asarray(u))), np.zeros(3), basis, eps)
        finite = bool(np.isfinite(jac_np).all())
        candidates = candidate_cpu = None
        if finite:
            direction = jac_np[0] / max(float(np.linalg.norm(jac_np[0])), 1e-12)
            rng = np.random.default_rng(20261007)
            random_directions = rng.normal(size=(6, 3))
            random_directions /= np.linalg.norm(random_directions, axis=1)[:, None]
            candidates = np.vstack([np.zeros(3), direction, -direction, random_directions]) * 0.03
            candidate_cpu = np.vstack([cpu_metrics(delta) for delta in candidates])
        result["gradient"] = {
            "controls": [f"{finger}_yaw" for finger in d1.FINGERS], "metric_names": list(d1.METRIC_NAMES),
            "finite_difference_step_rad": eps,
            "ad_jacobian": jac_np.tolist() if finite else None,
            "ad_nonfinite_count": int(np.size(jac_np) - np.isfinite(jac_np).sum()),
            "mjx_fd_jacobian": mjx_fd.tolist(),
            "cpu_fd_jacobian": cpu_fd.tolist(),
            "mjx_base_metrics": mjx_base.tolist(), "cpu_base_metrics": cpu_base.tolist(),
            "jacobian_first_s": jac_first_s, "jacobian_warm_s": jac_warm_s,
            "cpu_fd_7_evals_s": cpu_fd_s,
            "candidate_status": "evaluated" if finite else "skipped_nonfinite_ad",
            "candidate_delta_rad": None if candidates is None else candidates.tolist(),
            "candidate_cpu_metrics": None if candidate_cpu is None else candidate_cpu.tolist(),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
