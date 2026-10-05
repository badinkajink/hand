#!/usr/bin/env python3
"""Articulated SR2 torsional transfer and backend throughput/fidelity benchmark.

Fixed palm, all nine calibrated finger joints, two spherical caps, free tool.
The tool starts at the pinch centre (no gravity); shared position/rotation guide
forces prevent unrelated long-range drift. Input torque is identical open loop.
This isolates contact transfer, rather than retesting pickup success.
"""

import argparse
import json
import math
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import hom_chain as C
import hom_hand_brake as B
from contact_surface.geometry import Surface
from contact_surface.metrics import contacts
from contact_surface.records import CsvLog, ROOT, versions, write_json
from contact_surface.runtime import step1
from contact_surface.scaling import compensated_solref

DOC = ROOT / "docs/experiments/20261004-codex"


def build(config):
    spacing = config.get("spacing_mm", 1.0)
    policy = config.get("policy", "runtime")
    tr = 0.03
    # Direct format can express the desired k even when positive-format d cannot.
    p = Surface(spacing_mm=spacing, relaxation=0.06).parameters()
    p.update(
        relaxation=tr,
        mapping_policy="runtime_selected_direct" if policy.startswith("runtime") else "compiled",
    )
    p["tangent_damping_ratio"] = config.get("tangent_damping_ratio", 10.0)
    qdict, err = C.postures([0.0])
    qtouch = qdict[0.0]
    dirs = C.contact_dirs(qtouch)
    trial = C.make_trial(0, d_cg=0.0, perturb=False)
    trial["mscale"] = config.get("mass_scale", 1.0)
    spec = f"mj:spheres:s{spacing}:rs0.75:tr0.03:ir100"
    xml, info, _ = C.chain_scene(spec, trial, dirs, pad_tc=(0.015, 0.9))
    tree = ET.fromstring(xml)
    wb = tree.find("worldbody")
    for el in list(wb):
        if el.tag == "geom" or (el.tag == "body" and el.get("name") not in ("palm_pose", "tool")):
            wb.remove(el)
    palm = tree.find(".//body[@name='palm_pose']")
    palm.set("pos", "0 0 0.3")
    for el in list(palm):
        if el.tag == "joint" and el.get("name", "").startswith("palm_"):
            palm.remove(el)
    act = tree.find("actuator")
    for el in list(act):
        if el.get("joint", "").startswith("palm_"):
            act.remove(el)
    P, u, a = C.pinch_frame()
    P = P + [0, 0, 0.3]
    tool = tree.find(".//body[@name='tool']")
    tool.set("pos", " ".join(map(str, P)))
    d0 = 0.9 if policy in ("compiled", "runtime_standard") else 0.0001
    ir = 100.0 if policy in ("compiled", "runtime_standard") else 10000.0
    opt = tree.find("option")
    opt.set("timestep", str(config.get("timestep", 5e-5)))
    opt.set("gravity", "0 0 0")
    opt.set("jacobian", "dense")
    opt.set("impratio", str(ir))
    opt.set("integrator", "implicitfast")
    flag = opt.find("flag")
    if flag is None:
        flag = ET.SubElement(opt, "flag")
    if policy.startswith("runtime") and policy != "runtime_approx":
        flag.set("diagexact", "enable")
    contact = tree.find("contact")
    if contact is None:
        contact = ET.SubElement(tree, "contact")
    for geom in tree.iter("geom"):
        name = geom.get("name", "")
        is_pad = name.startswith(("thumb_pad", "index_pad"))
        geom.set("contype", "1" if is_pad else ("2" if name == "tool" else "0"))
        geom.set("conaffinity", "2" if is_pad else ("1" if name == "tool" else "0"))
        if is_pad:
            geom.set("solimp", f"{d0} {d0} .001 .5 2")
            ET.SubElement(
                contact,
                "pair",
                geom1=name,
                geom2="tool",
                condim="3",
                friction="1 1 0 0 0",
                solimp=f"{d0} {d0} .001 .5 2",
                solref="-1 -1",
                solreffriction="0 -1",
            )
    xml = ET.tostring(tree, encoding="unicode")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    fq = np.array([m.jnt_qposadr[m.joint(n).id] for f in B.FINGERS for n in B.JOINTS[f]])
    fv = np.array([m.jnt_dofadr[m.joint(n).id] for f in B.FINGERS for n in B.JOINTS[f]])
    d.qpos[fq] = qtouch
    mujoco.mj_forward(m, d)
    jacp = np.zeros((3, m.nv))
    targets = qtouch.copy()
    for j, (f, sign) in enumerate((("thumb", 1.0), ("index", -1.0))):
        mujoco.mj_jacBody(m, d, jacp, None, m.body(f + "_tip").id)
        targets[j * 3 : j * 3 + 3] += (
            jacp[:, fv[j * 3 : j * 3 + 3]].T
            @ (sign * config.get("pad_load", 2.0) * u)
            / B.PLANT["kp"]
        )
    d.ctrl[:] = targets
    by_geom = {}
    for i in range(m.npair):
        g = int(m.pair_geom1[i])
        if not m.geom(g).name.startswith(("thumb_pad", "index_pad")):
            g = int(m.pair_geom2[i])
        by_geom[g] = p
        lam = sum(
            m.body_invweight0[m.geom_bodyid[int(x)], 0] for x in [m.pair_geom1[i], m.pair_geom2[i]]
        )
        sr = compensated_solref(p["stiffness"], tr, d0, lam)
        m.pair_solref[i] = sr
        m.pair_solreffriction[i] = [0.0, sr[1] * p["tangent_damping_ratio"] / ir]
        m.geom_solref[g] = sr
    p["sample_parameters_by_geom"] = by_geom
    stiffness_by_geom = np.zeros(m.ngeom)
    stiffness_by_geom[list(by_geom)] = p["stiffness"]
    p["sample_stiffness_by_geom"] = stiffness_by_geom
    mujoco.mj_forward(m, d)
    tool_id = m.body("tool").id
    j = m.body_jntadr[tool_id]
    meta = dict(
        P=P.tolist(),
        u=u.tolist(),
        a=a.tolist(),
        tool=tool_id,
        tqa=int(m.jnt_qposadr[j]),
        tva=int(m.jnt_dofadr[j]),
        targets=targets.tolist(),
        qtouch=qtouch.tolist(),
        fq=fq.tolist(),
        ik_error_m=err,
        n_samples_per_pad=p["count"],
        k_i=p["stiffness"],
        sum_k_per_pad=p["count"] * p["stiffness"],
        sum_area_per_pad=p["count"] * p["nominal_area"],
        tool_mass_kg=float(m.body_mass[tool_id]),
        guide_stiffness=config.get("guide_stiffness", 20.0),
        payload_gravity=config.get("payload_gravity", 0.0),
        d0=d0,
        impratio=ir,
        tangent_damping_ratio=p["tangent_damping_ratio"],
        initial_qpos=d.qpos.tolist(),
    )
    return m, d, p, meta


def command(t, config):
    """Two opposing raised-cosine torque pulses, with final free decay."""
    tau = config.get("torque_peak", 0.003)
    if config.get("scenario") == "holding_ramp":
        return tau * min(1.0, max(0.0, (t - 0.3) / 1.0))
    if 0.3 <= t < 0.45:
        return tau * math.sin(math.pi * (t - 0.3) / 0.15) ** 2
    if 0.45 <= t < 0.6:
        return -tau * math.sin(math.pi * (t - 0.45) / 0.15) ** 2
    return 0.0


def wrench(m, d, meta, tau):
    """Same linear centre guide and rotation damping for every backend."""
    qa, va, bid = meta["tqa"], meta["tva"], meta["tool"]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, d.qpos[qa + 3 : qa + 7])
    omega = R.reshape(3, 3) @ d.qvel[va + 3 : va + 6]
    force = (
        meta.get("guide_stiffness", 20.0) * (np.array(meta["P"]) - d.qpos[qa : qa + 3])
        - 0.05 * d.qvel[va : va + 3]
    )
    # Separate payload test: ramp object-only weight after the unloaded settle.
    if meta.get("payload_gravity", 0.0):
        force[2] -= (
            m.body_mass[bid] * meta["payload_gravity"] * np.clip((d.time - 0.3) / 0.1, 0.0, 1.0)
        )
    torque = tau * np.array(meta["u"]) - 0.00002 * omega
    d.xfrc_applied[bid, :3] = force
    d.xfrc_applied[bid, 3:] = torque
    return omega


def run_cpu(config, directory, on_frame=None):
    m, d, p, meta = build(config)
    mujoco.mj_saveLastXML(str(directory / "model.xml"), m)
    write_json(directory / "fixture.json", meta)
    dt, duration = m.opt.timestep, config.get("duration", 0.8)
    # Independent constitutive audit includes viscous force and unilateral clipping.
    sums = np.zeros(4)
    timings, trace, contact_count, err_frames, velocities = [], [], [], [], []
    coefficient_max, peak_displacement, joint_motion, max_iterations = 0.0, 0.0, 0.0, 0
    status, failure = "complete", None
    fq = np.array(meta["fq"])
    start = time.perf_counter()
    stride = max(1, round(0.001 / dt))
    if on_frame is not None:
        on_frame(m, d, meta, [], {"time": float(d.time), "stage": "initial"})
    with CsvLog(directory / "contacts.csv.gz") as log:
        for k in range(round(duration / dt)):
            t = float(d.time)
            tick = time.perf_counter()
            omega = wrench(m, d, meta, command(t, config))
            if config.get("policy", "runtime").startswith("runtime"):
                step1(m, d, p)
                mujoco.mj_step2(m, d)
            else:
                mujoco.mj_step(m, d)
            timings.append(time.perf_counter() - tick)
            displacement = float(np.linalg.norm(d.qpos[meta["tqa"] : meta["tqa"] + 3] - meta["P"]))
            peak_displacement = max(peak_displacement, displacement)
            joint_motion = max(joint_motion, float(np.max(np.abs(d.qpos[fq] - meta["qtouch"]))))
            max_iterations = max(max_iterations, int(max(d.solver_niter)))
            if (
                not np.isfinite(d.qpos).all()
                or displacement > 0.025
                or np.max(np.abs(d.qvel)) > 2000
            ):
                status, failure = "failed", "nonfinite/25 mm escape/2000 rad/s robustness bound"
                if on_frame is not None:
                    on_frame(
                        m,
                        d,
                        meta,
                        [],
                        {"time": float(d.time), "stage": "stopped", "failure": failure},
                    )
                break
            # Constitutive residual on EVERY physics step; full records at 1 kHz.
            if t >= 0.3 and d.ncon:
                adrs = np.array(d.contact.efc_address[: d.ncon])
                valid = adrs >= 0
                adrs = adrs[valid]
                delta = np.maximum(0.0, -np.array(d.contact.dist[: d.ncon])[valid])
                ref = np.maximum(0.0, p["stiffness"] * (delta - p["relaxation"] * d.efc_vel[adrs]))
                sums[:2] += [float(np.sum(np.abs(d.efc_force[adrs] - ref))), float(np.sum(ref))]
            # Audit at 1 kHz; stepping/timing contains no force logging or matrix inversion.
            if k % stride:
                continue
            rows = contacts(m, d, p, t)
            Fn, Ftau, Fn_ref, nsliding = {"thumb": 0.0, "index": 0.0}, 0.0, 0.0, 0
            torsion_capacity = 0.0
            frame_num, frame_den = 0.0, 0.0
            for r in rows:
                expected = max(
                    0.0, r["k_i"] * (r["penetration"] - p["relaxation"] * r["v_contact_0"])
                )
                difference = r["f_n"] - expected
                r.update(physical_force=expected, dynamic_residual=difference)
                log.row(r)
                Fn[r["fingertip"]] += r["f_n"]
                pos = np.array([r[f"position_{i}"] for i in range(3)])
                force = np.array([r[f"force_{i}"] for i in range(3)])
                normal = np.array([r[f"normal_{i}"] for i in range(3)])
                vunit = np.cross(meta["u"], pos - meta["P"])
                tangent_unit = vunit - np.dot(vunit, normal) * normal
                torsion_capacity += r["friction_0"] * r["f_n"] * np.linalg.norm(tangent_unit)
                Ftau += float(np.cross(pos - meta["P"], force) @ meta["u"])
                Fn_ref += expected
                frame_num += abs(difference)
                frame_den += expected
                coefficient_max = max(
                    coefficient_max, abs(r["stiffness_predicted"] / p["stiffness"] - 1)
                )
                nsliding += r["slip_speed"] > 0.001
            if t >= 0.3:
                sums[2:] += [abs(Ftau), sum(Fn.values())]
                if frame_den > 1e-9:
                    err_frames.append(frame_num / frame_den)
            contact_count.append(len(rows))
            velocities.append(float(omega @ meta["u"]))
            trace.append(
                dict(
                    time=t,
                    torque=command(t, config),
                    normal_thumb=Fn["thumb"],
                    normal_index=Fn["index"],
                    reference_normal=Fn_ref,
                    contact_torque=Ftau,
                    angular_velocity=float(omega @ meta["u"]),
                    contacts=len(rows),
                    sliding=nsliding,
                    displacement_m=displacement,
                    relative_L1=frame_num / frame_den if frame_den else None,
                    solver_niter=int(max(d.solver_niter)),
                    pressure_torsion_capacity_Nm=float(torsion_capacity),
                )
            )
            if on_frame is not None:
                on_frame(m, d, meta, rows, trace[-1])
    write_json(directory / "trace.json", trace)
    breakaway = {}
    for threshold in (0.05, 0.1, 1.0):
        # Operational threshold sustained for 20 ms; log inertia-dependent lag.
        for i in range(len(trace) - 19):
            window = trace[i : i + 20]
            if window[0]["time"] >= 0.3 and all(
                abs(x["angular_velocity"]) >= threshold for x in window
            ):
                breakaway[str(threshold)] = dict(
                    torque=window[0]["torque"],
                    time=window[0]["time"],
                    normal_total=window[0]["normal_thumb"] + window[0]["normal_index"],
                )
                break
    return dict(
        status=status,
        failure=failure,
        backend="mujoco_cpu",
        config=config,
        fixture=meta,
        simulated_s=float(d.time),
        wall_s=time.perf_counter() - start,
        physics_ms_per_step_median=float(np.median(timings) * 1000),
        physics_s=sum(timings),
        real_time_factor_physics=float(d.time / sum(timings)),
        normal_L1_relative=float(sums[0] / sums[1]) if sums[1] else None,
        frame_L1_p95=float(np.quantile(err_frames, 0.95)) if err_frames else None,
        coefficient_relative_max=coefficient_max,
        peak_angular_speed=max(map(abs, velocities), default=0),
        peak_displacement_mm=peak_displacement * 1000,
        joint_motion_max_rad=joint_motion,
        mean_contacts=float(np.mean(contact_count)),
        peak_contacts=max(contact_count, default=0),
        solver_iterations_max=max_iterations,
        steady_pad_load=trace[min(round(0.29 / 0.001), len(trace) - 1)] if trace else None,
        audit_hz=1000.0,
        breakaway=breakaway,
        residual_audit_hz=1.0 / dt,
        mean_torque_arm_m=float(sums[2] / sums[3]) if sums[3] else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DOC / "configs/transfer.json")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/20261004-distributed-contact/transfer"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case in spec["cases"]:
        c = dict(spec.get("defaults", {}), **case)
        rid = c.pop("name")
        manifest.append(dict(run_id=rid, config=c))
    if (args.out / "manifest.json").exists() and json.loads(
        (args.out / "manifest.json").read_text()
    ) != manifest:
        raise ValueError("Manifest changed: choose a new output directory")
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "versions.json", versions())
    launched = 0
    for item in manifest:
        directory = args.out / item["run_id"]
        if (directory / "summary.json").exists():
            continue
        if args.limit is not None and launched >= args.limit:
            break
        directory.mkdir(exist_ok=True)
        write_json(directory / "config.json", item["config"])
        try:
            s = run_cpu(item["config"], directory)
        except Exception as e:
            s = dict(
                status="failed",
                error=repr(e),
                traceback=traceback.format_exc(),
                config=item["config"],
            )
        write_json(directory / "summary.json", s)
        print(item["run_id"], s["status"], s.get("normal_L1_relative"), flush=True)
        launched += 1
    summaries = [
        dict(
            run_id=x["run_id"], **json.loads((args.out / x["run_id"] / "summary.json").read_text())
        )
        for x in manifest
        if (args.out / x["run_id"] / "summary.json").exists()
    ]
    write_json(args.out / "summary.json", summaries)


if __name__ == "__main__":
    main()
