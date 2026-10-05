#!/usr/bin/env python3
"""Newton hydroelastic SR2 transfer; native mapping and explicit force-space adapter.

Pressure law kh=E/R, rigid-tool limit (kh_tool=1e4 kh_pad), zero margins/gaps,
mu=1 and no phenomenological point torsion. Translation modifies native solref
before constraint construction; it never overwrites a solved force. The pinned
GPU stack uses compiled body inverse weights, not CPU diagexact.
"""

import argparse
import json
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import newton
import numpy as np
import warp as wp
from newton.geometry import HydroelasticSDF

from contact_surface.records import ROOT, versions, write_json
from contact_surface.scaling import physical_coefficients
from distributed_contact_transfer import build

vec5 = wp.types.vector(length=5, dtype=wp.float32)


@wp.kernel
def convert_force_space(
    n: wp.array[int],
    geom: wp.array[wp.vec2i],
    world: wp.array[int],
    bodyid: wp.array[int],
    invw: wp.array2d[wp.vec2f],
    solref: wp.array[wp.vec2f],
    solimp: wp.array[vec5],
    fricref: wp.array[wp.vec2f],
    irinv: wp.array[float],
):
    i = wp.tid()
    if i >= n[0]:
        return
    sr = solref[i]
    imp = solimp[i][1]
    # Native converter encodes the Newton per-face secant stiffness without
    # the inverse-weight factor (see pinned kernels.py lines 605 onward).
    if sr[0] <= 0.0 or sr[1] <= 0.0:
        return
    k = 1.0 / (sr[0] * sr[0] * sr[1] * sr[1] * (1.0 - imp))
    g = geom[i]
    w = world[i]
    lam = invw[w, bodyid[g[0]]][0] + invw[w, bodyid[g[1]]][0]
    s = (1.0 - 0.0001) * lam
    solref[i] = wp.vec2f(-k * s, -k * 0.03 * s)
    solimp[i] = vec5(0.0001, 0.0001, 0.001, 0.5, 2.0)
    fricref[i] = wp.vec2f(0.0, -k * 0.03 * s * 10.0 * irinv[w] * irinv[w])


class TranslatedSolver(newton.solvers.SolverMuJoCo):
    def _convert_contacts_to_mjwarp(self, model, state, contacts):
        super()._convert_contacts_to_mjwarp(model, state, contacts)
        m, d = self.mjw_model, self.mjw_data
        wp.launch(
            convert_force_space,
            dim=d.naconmax,
            inputs=[
                d.nacon,
                d.contact.geom,
                d.contact.worldid,
                m.geom_bodyid,
                m.body_invweight0,
                d.contact.solref,
                d.contact.solimp,
                d.contact.solreffriction,
                m.opt.impratio_invsqrt,
            ],
        )


@wp.kernel
def drive(
    bodyq: wp.array[wp.transform],
    bodyv: wp.array[wp.spatial_vector],
    bodyf: wp.array[wp.spatial_vector],
    times: wp.array[float],
    tool: int,
    center: wp.vec3,
    axis: wp.vec3,
    peak: float,
):
    t = times[0]
    tau = float(0.0)
    if t >= 0.3 and t < 0.45:
        tau = peak * wp.pow(wp.sin(wp.pi * (t - 0.3) / 0.15), 2.0)
    elif t >= 0.45 and t < 0.6:
        tau = -peak * wp.pow(wp.sin(wp.pi * (t - 0.45) / 0.15), 2.0)
    position = wp.transform_get_translation(bodyq[tool])
    force = 20.0 * (center - position) - 0.05 * wp.spatial_top(bodyv[tool])
    torque = tau * axis - 0.00002 * wp.spatial_bottom(bodyv[tool])
    bodyf[tool] = wp.spatial_vector(force, torque)


def setup(config, directory):
    m, d, p, meta = build(dict(policy="compiled", spacing_mm=2.0, timestep=config["timestep"]))
    mujoco.mj_saveLastXML(str(directory / "source.xml"), m)
    tree = ET.parse(directory / "source.xml").getroot()
    for parent in tree.iter():
        for el in list(parent):
            if el.tag == "geom" and el.get("name", "").startswith(("thumb_pad", "index_pad")):
                parent.remove(el)
    contact = tree.find("contact")
    if contact is not None:
        tree.remove(contact)
    for geom in tree.iter("geom"):
        name = geom.get("name", "")
        if name in ("thumb_tipsphere", "index_tipsphere"):
            geom.set("contype", "1")
            geom.set("conaffinity", "2")
            geom.set("solimp", ".9 .9 .001 .5 2")
            geom.set("friction", "1 0 0")
    xml = ET.tostring(tree, encoding="unicode")
    (directory / "hand_sphere.xml").write_text(xml)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_mjcf(xml, ctrl_direct=True, parse_sites=False, parse_visuals=False)
    qstart = meta["qtouch"]
    if config.get("static_squeeze_m"):
        import hom_chain as C

        qstart = C.postures([-config["static_squeeze_m"]])[0][-config["static_squeeze_m"]]
    for f, qq in zip(("thumb", "index", "middle"), np.array(qstart).reshape(3, 3)):
        for name, q in zip((f + "_yaw", f + "_mcp", f + "_pip"), qq):
            j = next(i for i, x in enumerate(builder.joint_label) if x.split("/")[-1] == name)
            builder.joint_q[builder.joint_q_start[j]] = float(q)
    shapes = []
    for i, label in enumerate(builder.shape_label):
        name = label.split("/")[-1]
        if name not in ("thumb_tipsphere", "index_tipsphere", "tool"):
            continue
        shapes.append(i)
        builder.shape_flags[i] |= int(newton.ShapeFlags.HYDROELASTIC)
        builder.shape_sdf_target_voxel_size[i] = config.get("voxel_mm", 0.5) * 0.001
        builder.shape_sdf_narrow_band_range[i] = (-0.003, 0.003)
        builder.shape_sdf_padding[i] = 0.003
        builder.shape_margin[i] = 0.0
        builder.shape_gap[i] = 0.0
        builder.shape_material_kh[i] = (
            1e7 / 0.01055 * (config.get("tool_stiffness_ratio", 1e4) if name == "tool" else 1.0)
        )
        builder.shape_material_mu[i] = 1.0
        builder.shape_material_kf[i] = config.get("friction_gain", 1000.0)
        builder.shape_material_mu_torsional[i] = 0.0
        builder.shape_material_mu_rolling[i] = 0.0
    model = builder.finalize(device="cuda:0")
    hydro = HydroelasticSDF.Config(
        reduce_contacts=config.get("reduce_contacts", True),
        anchor_contact=True,
        moment_matching=config.get("moment_matching", False),
    )
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=config.get("reduce_contacts", True),
        rigid_contact_max=2048,
        broad_phase="explicit",
        sdf_hydroelastic_config=hydro,
    )
    if config.get("pressure_only"):
        s0 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        cc = pipeline.contacts()
        pipeline.collide(s0, cc)
        d.qpos[np.array(meta["fq"])] = qstart
        mujoco.mj_kinematics(m, d)
        pose_errors = {}
        poses = s0.body_q.numpy()
        for name in ("thumb_tip", "index_tip", "tool"):
            bid = next(i for i, x in enumerate(model.body_label) if x.split("/")[-1] == name)
            pose_errors[name] = float(np.linalg.norm(poses[bid, :3] - d.xpos[m.body(name).id]))
        if max(pose_errors.values()) > 1e-6:
            raise ValueError(f"Geometry import parity: {pose_errors}")
        meta["initial_pose_errors_m"] = pose_errors
        return model, None, pipeline, s0, None, None, cc, meta
    cls = (
        TranslatedSolver
        if config.get("mapping", "native") == "translated"
        else newton.solvers.SolverMuJoCo
    )
    solver = cls(
        model,
        use_mujoco_contacts=False,
        disable_sensors=True,
        njmax=8192,
        nconmax=2048,
        iterations=200,
        ls_iterations=50,
        cone="elliptic",
        jacobian="dense",
        integrator="implicitfast",
        tolerance=1e-8,
        impratio=config.get("impratio", 100.0),
    )
    s0, s1 = model.state(), model.state()
    ctrl = model.control()
    ctrl.mujoco.ctrl.assign(np.array(meta["targets"], dtype=np.float32))
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    bid = next(i for i, x in enumerate(model.body_label) if x.split("/")[-1] == "tool")
    cc = pipeline.contacts()
    pipeline.collide(s0, cc)
    meta.update(
        newton_tool=bid,
        hydro_shapes=shapes,
        body_count=model.body_count,
        shape_count=model.shape_count,
        kh_pad=1e7 / 0.01055,
        kh_tool=config.get("tool_stiffness_ratio", 1e4) * 1e7 / 0.01055,
    )
    meta["inv_weights"] = solver.mjw_model.body_invweight0.numpy().tolist()
    return model, solver, pipeline, s0, s1, ctrl, cc, meta


def run(config, directory, on_frame=None):
    startup = time.perf_counter()
    model, solver, pipeline, s0, s1, ctrl, cc, meta = setup(config, directory)
    if on_frame is not None:
        on_frame(model, solver, s0, meta, [], {"time": 0.0, "stage": "initial"})
    dt = config["timestep"]
    # Compile/warm one physical step, then reconstruct initial states for capture.
    pipeline.collide(s0, cc)
    solver.step(s0, s1, ctrl, cc, dt)
    solver.reset(s0)
    solver.reset(s1)
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    solver.mjw_data.time.zero_()
    wp.launch(
        drive,
        dim=1,
        inputs=[
            s0.body_q,
            s0.body_qd,
            s0.body_f,
            solver.mjw_data.time,
            meta["newton_tool"],
            wp.vec3(meta["P"]),
            wp.vec3(meta["u"]),
            config.get("torque_peak", 0.012),
        ],
    )
    wp.synchronize()
    block = round(0.005 / dt)
    # Capture an even number of ping-pong steps so input/output identities agree.
    if block % 2:
        block += 1
    with wp.ScopedCapture() as cap:
        for _ in range(block):
            s0.clear_forces()
            wp.launch(
                drive,
                dim=1,
                inputs=[
                    s0.body_q,
                    s0.body_qd,
                    s0.body_f,
                    solver.mjw_data.time,
                    meta["newton_tool"],
                    wp.vec3(meta["P"]),
                    wp.vec3(meta["u"]),
                    config.get("torque_peak", 0.012),
                ],
            )
            pipeline.collide(s0, cc)
            solver.step(s0, s1, ctrl, cc, dt)
            s0, s1 = s1, s0
    solver.reset(s0)
    solver.reset(s1)
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    solver.mjw_data.time.zero_()
    wp.synchronize()
    startup = time.perf_counter() - startup
    elapsed, trace = 0.0, []
    status, failure = "complete", None
    nums, dens = [], []
    start = time.perf_counter()
    inv_weights = solver.mjw_model.body_invweight0.numpy()[0]
    for _ in range(round(config.get("duration", 0.8) / (block * dt))):
        wp.synchronize()
        tick = time.perf_counter()
        wp.capture_launch(cap.graph)
        wp.synchronize()
        elapsed += time.perf_counter() - tick
        pose = s0.body_q.numpy()[meta["newton_tool"]]
        velocity = s0.body_qd.numpy()[meta["newton_tool"]]
        displacement = float(np.linalg.norm(pose[:3] - meta["P"]))
        if (
            not np.isfinite(pose).all()
            or not np.isfinite(velocity).all()
            or displacement > 0.025
            or np.max(np.abs(velocity)) > 2000
        ):
            status, failure = "failed", "nonfinite/25 mm escape/2000 rad/s robustness bound"
            if on_frame is not None:
                on_frame(
                    model,
                    solver,
                    s0,
                    meta,
                    [],
                    {
                        "time": float(solver.mjw_data.time.numpy()[0]),
                        "stage": "stopped",
                        "failure": failure,
                    },
                )
            break
        mjw.get_data_into(solver.mj_data, solver.mj_model, solver.mjw_data)
        m, d = solver.mj_model, solver.mj_data
        f6 = np.zeros(6)
        fn = 0.0
        moment = 0.0
        pred = 0.0
        num = 0.0
        points = []
        for i, c in enumerate(d.contact[: d.ncon]):
            if c.efc_address < 0:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            lam = sum(inv_weights[m.geom_bodyid[g], 0] for g in c.geom)
            k, b = physical_coefficients(c.solref, c.solimp, lam, dt)
            target = max(0.0, k * max(0.0, -c.dist) - b * d.efc_vel[c.efc_address])
            pred += target
            num += abs(f6[0] - target)
            fn += f6[0]
            force = np.array(c.frame).reshape(3, 3).T @ f6[:3]
            if m.geom(c.geom[1]).name.split("/")[-1] != "tool":
                force = -force
            moment += float(np.cross(np.array(c.pos) - meta["P"], force) @ meta["u"])
            if on_frame is not None:
                points.append(
                    [
                        *np.array(c.pos).tolist(),
                        *np.array(c.frame[:3]).tolist(),
                        float(f6[0]),
                        m.geom(c.geom[0]).name,
                        m.geom(c.geom[1]).name,
                    ]
                )
        t = float(solver.mjw_data.time.numpy()[0])
        if t >= 0.3:
            nums.append(num)
            dens.append(pred)
        trace.append(
            dict(
                time=t,
                normal_total=fn,
                reference_normal=pred,
                contact_torque=moment,
                angular_velocity=float(velocity[3:] @ meta["u"]),
                contacts=int(d.ncon),
                displacement_mm=displacement * 1000,
                relative_L1=num / pred if pred else None,
                solver_niter=int(max(d.solver_niter)),
            )
        )
        if on_frame is not None:
            on_frame(model, solver, s0, meta, points, trace[-1])
        if not np.isfinite(pose).all() or displacement > 0.025 or np.max(np.abs(velocity)) > 2000:
            status, failure = "failed", "nonfinite/25 mm escape/2000 rad/s robustness bound"
            break
    write_json(directory / "trace.json", trace)
    write_json(directory / "fixture.json", meta)
    simulated = block * dt * len(trace)
    return dict(
        status=status,
        failure=failure,
        backend="newton_hydroelastic",
        config=config,
        fixture=meta,
        startup_s=startup,
        wall_s=time.perf_counter() - start,
        physics_s=elapsed,
        simulated_s=simulated,
        real_time_factor_physics=simulated / elapsed,
        physics_ms_per_step_mean=elapsed / (max(1, len(trace)) * block) * 1000,
        normal_L1_relative=sum(nums) / sum(dens) if sum(dens) else None,
        peak_angular_speed=max((abs(x["angular_velocity"]) for x in trace), default=0.0),
        peak_displacement_mm=max((x["displacement_mm"] for x in trace), default=0.0),
        peak_contacts=max((x["contacts"] for x in trace), default=0),
        raw_contact_count_final=int(cc.rigid_contact_count.numpy()[0]),
        steady_pad_load=trace[min(round(0.29 / (block * dt)), len(trace) - 1)] if trace else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/20261004-distributed-contact/newton_transfer"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configs = [
        dict(
            name="native_lowfrictiongain",
            voxel_mm=0.5,
            timestep=0.00005,
            mapping="native",
            friction_gain=10.0,
        ),
        dict(
            name="native_finite_tool",
            voxel_mm=0.5,
            timestep=0.00005,
            mapping="native",
            friction_gain=10.0,
            tool_stiffness_ratio=100.0,
        ),
        dict(
            name="translated_finite_tool",
            voxel_mm=0.5,
            timestep=0.00005,
            mapping="translated",
            tool_stiffness_ratio=100.0,
        ),
        dict(name="native_h05", voxel_mm=0.5, timestep=0.00005, mapping="native"),
        dict(name="translated_h05", voxel_mm=0.5, timestep=0.00005, mapping="translated"),
        dict(name="translated_h025", voxel_mm=0.25, timestep=0.00005, mapping="translated"),
        dict(name="translated_dt25", voxel_mm=0.5, timestep=0.000025, mapping="translated"),
        dict(
            name="translated_unreduced",
            voxel_mm=0.5,
            timestep=0.00005,
            mapping="translated",
            reduce_contacts=False,
        ),
        dict(
            name="translated_fast",
            voxel_mm=0.5,
            timestep=0.00005,
            mapping="translated",
            torque_peak=0.048,
        ),
    ]
    write_json(args.out / "manifest.json", configs)
    write_json(args.out / "versions.json", versions())
    launched = 0
    for c in configs:
        path = args.out / c["name"]
        if (path / "summary.json").exists():
            continue
        if args.limit is not None and launched >= args.limit:
            break
        path.mkdir(exist_ok=True)
        write_json(path / "config.json", c)
        try:
            s = run(c, path)
        except Exception as e:
            s = dict(status="failed", error=repr(e), traceback=traceback.format_exc(), config=c)
        write_json(path / "summary.json", s)
        print(c["name"], s["status"], s.get("error"), flush=True)
        launched += 1
    write_json(
        args.out / "summary.json",
        [
            json.loads((args.out / c["name"] / "summary.json").read_text())
            for c in configs
            if (args.out / c["name"] / "summary.json").exists()
        ],
    )


if __name__ == "__main__":
    main()
