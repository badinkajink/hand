#!/usr/bin/env python3
"""mjlab Simulation (pinned 1.2 / MuJoCo Warp 3.6) matched hand benchmark.

The compiled physical mapping is supported by both CPU and GPU. The CPU-only
current-diagonal adapter is deliberately not claimed to run in the pinned stack.
CUDA graph timing includes the common applied-wrench kernel and native stepping;
compilation, allocation, instrumentation and host copies are reported separately.
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from mjlab.sim import Simulation, SimulationCfg, MujocoCfg

from contact_surface.records import ROOT, versions, write_json
from distributed_contact_transfer import build
from contact_surface.scaling import compensated_solref


@wp.kernel
def guide(
    qpos: wp.array2d[float],
    qvel: wp.array2d[float],
    times: wp.array[float],
    xfrc: wp.array2d[wp.spatial_vector],
    qa: int,
    va: int,
    body: int,
    center: wp.vec3,
    axis: wp.vec3,
    peak: float,
):
    w = wp.tid()
    t = times[w]
    tau = float(0.0)
    if t >= 0.3 and t < 0.45:
        tau = peak * wp.pow(wp.sin(wp.pi * (t - 0.3) / 0.15), 2.0)
    elif t >= 0.45 and t < 0.6:
        tau = -peak * wp.pow(wp.sin(wp.pi * (t - 0.45) / 0.15), 2.0)
    q = wp.quat(qpos[w, qa + 4], qpos[w, qa + 5], qpos[w, qa + 6], qpos[w, qa + 3])
    omega = wp.quat_rotate(q, wp.vec3(qvel[w, va + 3], qvel[w, va + 4], qvel[w, va + 5]))
    force = 20.0 * (
        center - wp.vec3(qpos[w, qa], qpos[w, qa + 1], qpos[w, qa + 2])
    ) - 0.05 * wp.vec3(qvel[w, va], qvel[w, va + 1], qvel[w, va + 2])
    torque = tau * axis - 0.00002 * omega
    xfrc[w, body] = wp.spatial_vector(force, torque)


def run(config, directory):
    m, d, p, meta = build(config)
    # Pinned Warp 3.6 scales elliptic rows using welded-body inverse weights.
    # CPU 3.6 uses geom-body weights. Translate the SAME physical k,c for each.
    for i in range(m.npair):
        lam = sum(
            m.body_invweight0[m.body_weldid[m.geom_bodyid[g]], 0]
            for g in (m.pair_geom1[i], m.pair_geom2[i])
        )
        sr = compensated_solref(p["stiffness"], p["relaxation"], meta["d0"], lam)
        m.pair_solref[i] = sr
        m.pair_solreffriction[i] = [0.0, sr[1] * 10.0 / m.opt.impratio]
    dt = m.opt.timestep
    nworld = config["nworld"]
    cfg = SimulationCfg(
        nconmax=2048,
        njmax=8192,
        mujoco=MujocoCfg(
            timestep=dt,
            integrator="implicitfast",
            impratio=m.opt.impratio,
            cone="elliptic",
            jacobian="dense",
            solver="newton",
            iterations=200,
            tolerance=config.get("solver_tolerance", 1e-8),
            ls_iterations=50,
            gravity=(0.0, 0.0, 0.0),
        ),
    )
    startup = time.perf_counter()
    sim = Simulation(nworld, cfg, m, "cuda:0")
    wd, wm = sim.wp_data, sim.wp_model
    wd.qpos.assign(np.tile(d.qpos, (nworld, 1)).astype(np.float32))
    wd.qvel.zero_()
    wd.qacc_warmstart.zero_()
    wd.time.zero_()
    wd.ctrl.assign(np.tile(d.ctrl, (nworld, 1)).astype(np.float32))
    sim.forward()
    wp.synchronize()
    block = round(0.01 / dt)
    inp = [
        wd.qpos,
        wd.qvel,
        wd.time,
        wd.xfrc_applied,
        meta["tqa"],
        meta["tva"],
        meta["tool"],
        wp.vec3(meta["P"]),
        wp.vec3(meta["u"]),
        config.get("torque_peak", 0.012),
    ]
    # Warm compilation before capture; then restore the exact initial state.
    wp.launch(guide, dim=nworld, inputs=inp)
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        for _ in range(block):
            wp.launch(guide, dim=nworld, inputs=inp)
            mjw.step(wm, wd)
    wd.qpos.assign(np.tile(d.qpos, (nworld, 1)).astype(np.float32))
    wd.qvel.zero_()
    wd.qacc_warmstart.zero_()
    wd.time.zero_()
    sim.forward()
    wp.synchronize()
    startup = time.perf_counter() - startup
    ticks, trace = [], []
    nums, dens = [], []
    status, failure = "complete", None
    start = time.perf_counter()
    for _ in range(round(config.get("duration", 0.8) / 0.01)):
        wp.synchronize()
        tick = time.perf_counter()
        wp.capture_launch(cap.graph)
        wp.synchronize()
        ticks.append(time.perf_counter() - tick)
        allq, allv = wd.qpos.numpy(), wd.qvel.numpy()
        displacement_all = np.linalg.norm(
            allq[:, meta["tqa"] : meta["tqa"] + 3] - meta["P"], axis=1
        )
        if (
            not np.isfinite(allq).all()
            or not np.isfinite(allv).all()
            or np.max(displacement_all) > 0.025
            or np.max(np.abs(allv)) > 2000
        ):
            status, failure = (
                "failed",
                "a batch world exceeded nonfinite/25 mm escape/2000 rad/s robustness bound",
            )
            break
        q = allq[0]
        v = allv[0]
        qa, va = meta["tqa"], meta["tva"]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q[qa + 3 : qa + 7].astype(float))
        omega = R.reshape(3, 3) @ v[va + 3 : va + 6]
        delta = float(np.linalg.norm(q[qa : qa + 3] - meta["P"]))
        mjw.get_data_into(d, m, wd, world_id=0)
        f6 = np.zeros(6)
        refsum, errsum = 0.0, 0.0
        fn = 0.0
        tau = 0.0
        for i, c in enumerate(d.contact[: d.ncon]):
            mujoco.mj_contactForce(m, d, i, f6)
            force = np.array(c.frame).reshape(3, 3).T @ f6[:3]
            # Pair geom1 is pad and geom2 tool in this shared fixture.
            if m.geom(c.geom[1]).name != "tool":
                force = -force
            fn += f6[0]
            if c.efc_address >= 0:
                target = max(
                    0.0,
                    p["stiffness"]
                    * (max(0.0, -c.dist) - p["relaxation"] * d.efc_vel[c.efc_address]),
                )
                refsum += target
                errsum += abs(f6[0] - target)
            tau += float(np.cross(np.array(c.pos) - meta["P"], force) @ meta["u"])
        if float(wd.time.numpy()[0]) >= 0.3:
            nums.append(errsum)
            dens.append(refsum)
        trace.append(
            dict(
                time=float(wd.time.numpy()[0]),
                qpos=q.tolist(),
                angular_velocity=float(omega @ meta["u"]),
                normal_total=fn,
                reference_normal=refsum,
                relative_L1=errsum / refsum if refsum else None,
                contact_torque=tau,
                contacts=int(d.ncon),
                displacement_mm=delta * 1000,
            )
        )
        if not np.isfinite(q).all() or delta > 0.025 or np.max(np.abs(v)) > 2000:
            status, failure = "failed", "nonfinite/25 mm escape/2000 rad/s robustness bound"
            break
    elapsed = sum(ticks)
    simulated = block * len(ticks) * dt
    write_json(directory / "trace.json", trace)
    return dict(
        status=status,
        failure=failure,
        backend="mjlab_simulation",
        config=config,
        fixture=meta,
        startup_s=startup,
        wall_s=time.perf_counter() - start,
        physics_s=elapsed,
        simulated_s=simulated,
        physics_ms_per_step_median=float(np.median(ticks) / block * 1000),
        aggregate_steps_per_s=nworld * block * len(ticks) / elapsed,
        real_time_factor_physics=simulated / elapsed,
        aggregate_real_time_factor=nworld * simulated / elapsed,
        peak_angular_speed=max(abs(x["angular_velocity"]) for x in trace),
        peak_displacement_mm=max(x["displacement_mm"] for x in trace),
        peak_contacts=max(x["contacts"] for x in trace),
        contact_capacity=2048,
        constraint_capacity=8192,
        normal_L1_sampled=sum(nums) / sum(dens) if sum(dens) else None,
        audit_hz=100.0,
        max_nefc=int(max(wd.nefc.numpy())),
        nacon_final=int(wd.nacon.numpy()[0]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/20261004-distributed-contact/mjlab_transfer"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configs = [
        dict(
            name=f"s{s}_batch{n}",
            spacing_mm=s,
            nworld=n,
            timestep=0.0005,
            policy="compiled",
            torque_peak=0.012,
            duration=0.8,
            solver_tolerance=1e-8,
        )
        for s in (1.0, 0.5)
        for n in (1, 32, 128)
    ]
    configs += [
        dict(
            name=f"fine_s{s}_batch{n}",
            spacing_mm=s,
            nworld=n,
            timestep=0.00005,
            policy="runtime_approx",
            torque_peak=0.012,
            duration=0.8,
            solver_tolerance=1e-8,
        )
        for s in (1.0, 0.5)
        for n in (1, 128)
    ]
    write_json(args.out / "manifest.json", configs)
    write_json(args.out / "versions.json", versions())
    launched = 0
    for config in configs:
        directory = args.out / config["name"]
        if (directory / "summary.json").exists():
            continue
        if args.limit is not None and launched >= args.limit:
            break
        directory.mkdir(exist_ok=True)
        write_json(directory / "config.json", config)
        try:
            summary = run(config, directory)
        except Exception as e:
            summary = dict(
                status="failed", error=repr(e), traceback=traceback.format_exc(), config=config
            )
        write_json(directory / "summary.json", summary)
        print(config["name"], summary["status"], summary.get("aggregate_steps_per_s"), flush=True)
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
