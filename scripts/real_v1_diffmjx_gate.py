#!/usr/bin/env python3
"""Check MJX contact dynamics at a replayed real_v1 grasp before optimizer studies.

The scene comes from the deployed D1 plan. The palm display plate is made
non-colliding because MJX 3.6 does not implement cylinder-box collisions;
the plate has no contacts in the replayed state. The tool, post and three
fingertips retain their original collision geometry and solver settings.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reorient_backends as rb  # noqa: E402


def make_scene(hand: str, numerics: str, impratio: float, no_post: bool = False,
               tool_capsule: bool = False, tip: str = "sphere",
               contact_model: str = "pt", pad_spacing_m: float = 0.001) -> tuple[Path, dict]:
    out_dir = ROOT / "assets/mjcf/experimental/20261007-diffmjx"
    scene, meta = rb.build_scene(hand, tip, contact_model, "fast", numerics, impratio,
                                 "scene", out_dir, pad_s=pad_spacing_m)
    root = ET.parse(scene).getroot()
    plate = root.find('.//body[@name="palm_pose"]/geom')
    assert plate is not None and plate.get("type") == "box"
    plate.set("contype", "0")
    plate.set("conaffinity", "0")
    if contact_model == "padsT":
        # The pad spheres keep tool contact. The mesh is visual in this local
        # probe; inter-finger mesh contact is checked against the full scene.
        for geom in root.iter("geom"):
            if geom.get("name", "").endswith("_tipvis"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
    if tool_capsule:
        tool_geom = root.find(f'.//body[@name="{rb.OBJ}"]/geom')
        assert tool_geom is not None and tool_geom.get("type") == "cylinder"
        tool_geom.set("type", "capsule")
    if no_post:
        post = root.find('.//body[@name="tool_platform"]/geom')
        assert post is not None
        post.set("contype", "0")
        post.set("conaffinity", "0")
    tag = "_mjx" + ("_capsule" if tool_capsule else "") + ("_no_post" if no_post else "")
    adapted = scene.with_name(scene.stem + tag + ".xml")
    ET.ElementTree(root).write(adapted)
    return adapted, meta


def grip_state(model, meta: dict, hand: str, seed: int):
    import mujoco

    data = mujoco.MjData(model)
    obj = model.body(rb.OBJ).id
    qadr = model.jnt_qposadr[model.body_jntadr[obj]]
    dx, dy, dyaw = rb.jitter(seed)
    data.qpos[qadr:qadr + 3] = np.asarray(meta["tool7"][:3]) + [dx, dy, 0.0]
    data.qpos[qadr + 3:qadr + 7] = rb._yaw_quat(meta["tool7"][3:], dyaw)
    for finger in rb.FINGERS:
        for joint in rb.JOINTS:
            name = f"{finger}_{joint}"
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = meta["q0"][name]
    plan, traj = rb.load_plan(hand)
    target = rb.schedule(plan, traj)[0][1]
    for name, value in target.items():
        data.ctrl[model.actuator("a_" + name).id] = value
    while data.time < rb.T_GRIP - 1e-9:
        mujoco.mj_step(model, data)
    return data


def clone_data(model, source):
    import mujoco

    data = mujoco.MjData(model)
    for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
        getattr(data, name)[:] = getattr(source, name)
    data.time = source.time
    mujoco.mj_forward(model, data)
    return data


def contact_summary(model, data) -> dict:
    import mujoco

    obj = model.body(rb.OBJ).id
    tool_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] == obj}
    force = {finger: 0.0 for finger in rb.FINGERS}
    min_dist = {finger: None for finger in rb.FINGERS}
    plate_contacts = 0
    wrench = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        g0, g1 = (int(v) for v in contact.geom)
        if g0 == 2 or g1 == 2:
            plate_contacts += 1
        if g0 not in tool_geoms and g1 not in tool_geoms:
            continue
        other = g1 if g0 in tool_geoms else g0
        body = model.body(model.geom_bodyid[other]).name
        for finger in rb.FINGERS:
            if body.startswith(finger + "_"):
                mujoco.mj_contactForce(model, data, i, wrench)
                force[finger] += max(0.0, float(wrench[0]))
                min_dist[finger] = float(contact.dist) if min_dist[finger] is None else min(
                    min_dist[finger], float(contact.dist))
    return {"finger_normal_force_N": force, "finger_min_dist_m": min_dist,
            "palm_plate_contacts": plate_contacts,
            "total_contacts": int(data.ncon)}


def mjx_contact_summary(model, data) -> dict:
    obj = model.body(rb.OBJ).id
    tool_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] == obj}
    contacts = data._impl.contact
    pairs = np.asarray(contacts.geom)
    distances = np.asarray(contacts.dist)
    addresses = np.asarray(contacts.efc_address)
    forces = np.asarray(data._impl.efc_force)
    out = {finger: {"n": 0, "normal_force_N": 0.0, "min_dist_m": None} for finger in rb.FINGERS}
    for (g0, g1), distance, address in zip(pairs, distances, addresses):
        g0, g1 = int(g0), int(g1)
        if g0 not in tool_geoms and g1 not in tool_geoms:
            continue
        other = g1 if g0 in tool_geoms else g0
        body = model.body(model.geom_bodyid[other]).name
        for finger in rb.FINGERS:
            if body.startswith(finger + "_"):
                row = out[finger]
                row["min_dist_m"] = float(distance) if row["min_dist_m"] is None else min(
                    row["min_dist_m"], float(distance))
                if 0 <= address < len(forces) and forces[address] > 1e-6:
                    row["n"] += 1
                    row["normal_force_N"] += float(forces[address])
    return out


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
    parser.add_argument("--out", type=Path, default=ROOT / "docs/experiments/20261007-diffmjx/20261007-mjx_gate.json")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    if args.contact_model in ("pads", "padsT") and args.tip != "tpu6":
        parser.error("sphere pads require --tip tpu6")
    scene, meta = make_scene(args.hand, args.numerics, args.impratio, args.no_post,
                             args.tool_capsule, args.tip, args.contact_model,
                             args.pad_spacing_mm * 0.001)
    model = mujoco.MjModel.from_xml_path(str(scene))
    if args.no_post or args.tool_capsule:
        supported_scene, _ = make_scene(args.hand, args.numerics, args.impratio,
                                        tip=args.tip, contact_model=args.contact_model,
                                        pad_spacing_m=args.pad_spacing_mm * 0.001)
        supported_model = mujoco.MjModel.from_xml_path(str(supported_scene))
        held = clone_data(model, grip_state(supported_model, meta, args.hand, args.seed))
    else:
        held = grip_state(model, meta, args.hand, args.seed)
    obj = model.body(rb.OBJ).id
    qadr = int(model.jnt_qposadr[model.body_jntadr[obj]])
    vadr = int(model.jnt_dofadr[model.body_jntadr[obj]])
    held_contacts = contact_summary(model, held)
    assert held_contacts["palm_plate_contacts"] == 0
    print("held state", held_contacts, flush=True)

    cpu = clone_data(model, held)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        mujoco.mj_step(model, cpu)
    cpu_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    mx = mjx.put_model(model)
    put_model_s = time.perf_counter() - t0
    print(f"put_model {put_model_s:.3f} s", flush=True)
    t0 = time.perf_counter()
    md = mjx.put_data(model, held)
    put_data_s = time.perf_counter() - t0
    print(f"put_data {put_data_s:.3f} s", flush=True)
    if jax.config.read("jax_enable_x64"):
        # put_data retains int32 contact indices; MJX recomputes them as int64
        # in x64 mode, which makes the scan carry fail its dtype check.
        md = md.tree_replace({
            "_impl.contact.geom1": md._impl.contact.geom1.astype(np.int64),
            "_impl.contact.geom2": md._impl.contact.geom2.astype(np.int64),
            "_impl.contact.geom": md._impl.contact.geom.astype(np.int64),
        })

    @jax.jit
    def forward(data):
        return mjx.forward(mx, data)

    jinitial = forward(md)
    jinitial.qpos.block_until_ready()
    print("mjx.forward compiled", flush=True)

    @jax.jit
    def rollout(data):
        def one(carry, _):
            return mjx.step(mx, carry), None
        final, _ = jax.lax.scan(one, data, None, length=args.steps)
        return final

    t0 = time.perf_counter()
    jdata = rollout(md)
    jdata.qpos.block_until_ready()
    mjx_first_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    jdata = rollout(md)
    jdata.qpos.block_until_ready()
    mjx_warm_s = time.perf_counter() - t0

    jpos = np.asarray(jdata.qpos)
    jvel = np.asarray(jdata.qvel)
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
        "put_model_s": put_model_s, "put_data_s": put_data_s,
        "initial_contact": held_contacts,
        "mjx_initial_contact": mjx_contact_summary(model, jinitial),
        "cpu_contact": contact_summary(model, cpu),
        "mjx_contact": mjx_contact_summary(model, jdata),
        "cpu_tool_qpos": cpu.qpos[qadr:qadr + 7].tolist(),
        "mjx_tool_qpos": jpos[qadr:qadr + 7].tolist(),
        "cpu_tool_qvel": cpu.qvel[vadr:vadr + 6].tolist(),
        "mjx_tool_qvel": jvel[vadr:vadr + 6].tolist(),
        "max_qpos_abs_error": float(np.max(np.abs(cpu.qpos - jpos))),
        "max_qvel_abs_error": float(np.max(np.abs(cpu.qvel - jvel))),
        "cpu_rollout_s": cpu_s, "mjx_first_s": mjx_first_s,
        "mjx_warm_s": mjx_warm_s,
    }

    if args.gradient:
        yaw_ids = np.array([model.actuator(f"a_{finger}_yaw").id for finger in rb.FINGERS])
        yaw_names = [f"{finger}_yaw" for finger in rb.FINGERS]

        @jax.jit
        def mjx_metrics(delta):
            state = md.replace(ctrl=md.ctrl.at[yaw_ids].add(delta))
            final = rollout(state)
            quat = final.qpos[qadr + 3:qadr + 7]
            vertical_cos = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
            return jnp.array([vertical_cos, final.qpos[qadr + 2],
                              final.qvel[vadr + 3], final.qfrc_constraint[vadr + 3]])

        def cpu_metrics(delta):
            state = clone_data(model, held)
            state.ctrl[yaw_ids] += delta
            for _ in range(args.steps):
                mujoco.mj_step(model, state)
            quat = state.qpos[qadr + 3:qadr + 7]
            vertical_cos = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
            return np.array([vertical_cos, state.qpos[qadr + 2],
                             state.qvel[vadr + 3], state.qfrc_constraint[vadr + 3]])

        zero = jnp.zeros(3)
        grad_fn = jax.jit(jax.jacfwd(mjx_metrics))
        t0 = time.perf_counter()
        jac = grad_fn(zero)
        jac.block_until_ready()
        jac_first_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        jac = grad_fn(zero)
        jac.block_until_ready()
        jac_warm_s = time.perf_counter() - t0
        jac_np = np.asarray(jac)
        eps = 0.01
        basis = np.eye(3)
        t0 = time.perf_counter()
        cpu_base = cpu_metrics(np.zeros(3))
        cpu_fd = np.column_stack([(cpu_metrics(eps * b) - cpu_metrics(-eps * b)) / (2 * eps)
                                  for b in basis])
        cpu_fd_s = time.perf_counter() - t0
        mjx_base = np.asarray(mjx_metrics(zero))
        mjx_fd = np.column_stack([(np.asarray(mjx_metrics(jnp.asarray(eps * b))) -
                                   np.asarray(mjx_metrics(jnp.asarray(-eps * b)))) / (2 * eps)
                                  for b in basis])
        candidate_status = "skipped_nonfinite_ad" if not np.isfinite(jac_np).all() else "evaluated"
        candidates = candidate_cpu = None
        if candidate_status == "evaluated":
            direction = jac_np[0]
            direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
            rng = np.random.default_rng(20261007)
            random_directions = rng.normal(size=(6, 3))
            random_directions /= np.linalg.norm(random_directions, axis=1)[:, None]
            candidates = np.vstack([np.zeros(3), direction, -direction, random_directions]) * 0.03
            candidate_cpu = np.vstack([cpu_metrics(delta) for delta in candidates])
        result["gradient"] = {
            "controls": yaw_names, "metric_names": ["tool_vertical_cos", "tool_z_m",
                                                  "tool_omega_x_rad_s", "tool_contact_torque_x_Nm"],
            "finite_difference_step_rad": eps,
            "ad_jacobian": np.where(np.isfinite(jac_np), jac_np, 0.0).tolist() if np.isfinite(jac_np).all() else None,
            "ad_nonfinite_count": int(np.size(jac_np) - np.isfinite(jac_np).sum()),
            "mjx_fd_jacobian": mjx_fd.tolist(),
            "cpu_fd_jacobian": cpu_fd.tolist(),
            "mjx_base_metrics": mjx_base.tolist(), "cpu_base_metrics": cpu_base.tolist(),
            "jacobian_first_s": jac_first_s, "jacobian_warm_s": jac_warm_s,
            "cpu_fd_7_evals_s": cpu_fd_s,
            "candidate_status": candidate_status,
            "candidate_delta_rad": None if candidates is None else candidates.tolist(),
            "candidate_cpu_metrics": None if candidate_cpu is None else candidate_cpu.tolist(),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
