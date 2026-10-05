#!/usr/bin/env python3
"""Matched fixed-palm SR2 torsion in Drake SAP hydroelastic contact.

Reuse the nine-joint source scene, calibrated servo targets, common applied
wrench and actual spherical fingertip geometry. No point torsion coefficients.
"""

import argparse
import json
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco
import hom_contact_rig as H
import hom_hand_brake as B
import hom_hand_drake as D
from contact_surface.records import ROOT, versions, write_json
from distributed_contact_transfer import build, command


def setup(config, path):
    m, d, p, meta = build(dict(config, policy="compiled", spacing_mm=2.0))
    mujoco.mj_saveLastXML(str(path / "source.xml"), m)
    root = ET.parse(path / "source.xml").getroot()
    for parent in root.iter():
        for el in list(parent):
            if el.tag != "geom":
                continue
            name = el.get("name", "")
            if name in ("thumb_tipsphere", "index_tipsphere"):
                el.set("contype", "1")
                el.set("conaffinity", "2")
            elif name != "tool":
                parent.remove(el)
    for c in root.findall("contact"):
        root.remove(c)
    xml = ET.tostring(root, encoding="unicode")
    (path / "hand_sphere.xml").write_text(xml)
    old_scene, old_dt = B.mj_scene, H.DT
    try:
        B.mj_scene = lambda *args, **kwargs: (xml, {})
        H.DT = config["timestep"]
        rig = D.DrakeHand(
            f"drake:hydro:rt0.03:hr{config.get('mesh_mm', 1.0)}",
            B.d8_offsets(),
            0.0,
            meta["qtouch"],
        )
    finally:
        B.mj_scene, H.DT = old_scene, old_dt
    rig.plant.mutable_gravity_field().set_gravity_vector(np.zeros(3))
    rig.set_targets(meta["targets"])
    # Verify the imported initial fingertip centres against the MJCF source.
    errors = {}
    for name in ["thumb_tip", "index_tip", "tool"]:
        pos = rig.plant.EvalBodyPoseInWorld(rig.pc, rig.plant.GetBodyByName(name)).translation()
        errors[name] = float(np.linalg.norm(pos - d.xpos[m.body(name).id]))
    meta["initial_pose_errors_m"] = errors
    if max(errors.values()) > 1e-6:
        raise ValueError(f"Geometry transfer failed: {errors}")
    return rig, meta


def run(config, path, on_frame=None):
    startup = time.perf_counter()
    rig, meta = setup(config, path)
    if on_frame is not None:
        on_frame(rig, meta, [], {"time": rig.t, "stage": "initial"})
    startup = time.perf_counter() - startup
    dt = config["timestep"]
    ticks = []
    trace = []
    status, failure = "complete", None
    start = time.perf_counter()
    for i in range(round(config.get("duration", 0.8) / dt)):
        t = rig.t
        tick = time.perf_counter()
        st = rig.tool_state()
        velocity = np.array(
            rig.plant.EvalBodySpatialVelocityInWorld(rig.pc, rig.toolb).translational()
        )
        force = 20.0 * (np.array(meta["P"]) - st["pos"]) - 0.05 * velocity
        torque = command(t, config) * np.array(meta["u"]) - 0.00002 * st["w"]
        rig.set_tool_wrench(force, torque)
        rig.step(dt)
        ticks.append(time.perf_counter() - tick)
        displacement = float(np.linalg.norm(st["pos"] - meta["P"]))
        if (
            not np.isfinite(st["pos"]).all()
            or displacement > 0.025
            or np.max(np.abs(st["w"])) > 2000
        ):
            status, failure = "failed", "nonfinite/25 mm escape/2000 rad/s robustness bound"
            if on_frame is not None:
                on_frame(rig, meta, [], {"time": rig.t, "stage": "stopped", "failure": failure})
            break
        if i % max(1, round(0.001 / dt)):
            continue
        cc = rig.contacts()
        cr = rig.plant.get_contact_results_output_port().Eval(rig.pc)
        moment = 0.0
        pressure_capacity = 0.0
        rescaled_capacity = 0.0
        faces = []
        capture_face = on_frame is not None and on_frame.due(rig.t)
        for j in range(cr.num_hydroelastic_contacts()):
            info = cr.hydroelastic_contact_info(j)
            srf = info.contact_surface()
            if srf.id_M() in rig.tool_gids:
                sign = 1.0
            elif srf.id_N() in rig.tool_gids:
                sign = -1.0
            else:
                continue
            F = sign * np.array(info.F_Ac_W().translational())
            T = sign * np.array(info.F_Ac_W().rotational())
            mesh = srf.tri_mesh_W() if srf.is_triangle() else srf.poly_mesh_W()
            field = srf.tri_e_MN() if srf.is_triangle() else srf.poly_e_MN()
            patch_capacity, patch_pressure = 0.0, 0.0
            for face in range(mesh.num_elements()):
                pos = np.array(mesh.element_centroid(face))
                pn = float(field.EvaluateCartesian(face, pos)) * mesh.area(face)
                vunit = np.cross(meta["u"], pos - meta["P"])
                patch_capacity += pn * np.linalg.norm(vunit)
                patch_pressure += pn
                if capture_face:
                    faces.append(
                        [
                            *pos.tolist(),
                            *np.array(mesh.face_normal(face)).tolist(),
                            pn,
                            float(field.EvaluateCartesian(face, pos)),
                            float(mesh.area(face)),
                        ]
                    )
            radial = np.array(srf.centroid()) - st["pos"]
            radial -= np.dot(radial, st["axis"]) * st["axis"]
            actual_normal = abs(np.dot(F, radial)) / max(np.linalg.norm(radial), 1e-12)
            pressure_capacity += patch_capacity
            rescaled_capacity += patch_capacity * actual_normal / max(patch_pressure, 1e-12)
            moment += float((T + np.cross(np.array(srf.centroid()) - meta["P"], F)) @ meta["u"])
        trace.append(
            dict(
                time=t,
                normal_total=cc["thumb"]["N"] + cc["index"]["N"],
                contact_torque=moment,
                angular_velocity=float(st["w"] @ meta["u"]),
                contacts=sum(x["n"] for x in cc.values()),
                displacement_mm=displacement * 1000,
                torque=command(t, config),
                pressure_torsion_capacity_Nm=pressure_capacity,
                rescaled_normal_torsion_capacity_Nm=rescaled_capacity,
            )
        )
        if capture_face:
            on_frame(rig, meta, faces, trace[-1])
    write_json(path / "trace.json", trace)
    breaks = {}
    for threshold in (0.05, 0.1, 1.0):
        for i in range(len(trace) - 19):
            w = trace[i : i + 20]
            if w[0]["time"] >= 0.3 and all(abs(x["angular_velocity"]) >= threshold for x in w):
                breaks[str(threshold)] = dict(
                    torque=w[0]["torque"], time=w[0]["time"], normal_total=w[0]["normal_total"]
                )
                break
    return dict(
        breakaway=breaks,
        status=status,
        failure=failure,
        backend="drake_hydroelastic",
        config=config,
        fixture=meta,
        startup_s=startup,
        wall_s=time.perf_counter() - start,
        physics_s=sum(ticks),
        simulated_s=rig.t,
        physics_ms_per_step_median=float(np.median(ticks) * 1000),
        real_time_factor_physics=rig.t / sum(ticks),
        peak_angular_speed=max(abs(x["angular_velocity"]) for x in trace),
        peak_displacement_mm=max(x["displacement_mm"] for x in trace),
        peak_contacts=max(x["contacts"] for x in trace),
        steady_pad_load=trace[min(290, len(trace) - 1)],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/20261004-distributed-contact/drake_transfer"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = [
        dict(name=f"drake_tau{t}", timestep=0.0005, torque_peak=t) for t in (0.003, 0.012, 0.048)
    ]
    cases += [
        dict(name="drake_dt1ms", timestep=0.001, torque_peak=0.012),
        dict(name="drake_dt01ms", timestep=0.0001, torque_peak=0.012),
        dict(name="drake_mesh05", timestep=0.0005, torque_peak=0.012, mesh_mm=0.5),
    ]
    cases += [
        dict(
            name=f"drake_hold_m{mass}",
            scenario="holding_ramp",
            duration=1.1,
            timestep=0.0005,
            torque_peak=0.008,
            mass_scale=mass,
        )
        for mass in (0.25, 1.0, 4.0)
    ]
    cases += [
        dict(
            name="drake_hold_load4",
            scenario="holding_ramp",
            duration=1.1,
            timestep=0.0005,
            torque_peak=0.015,
            pad_load=4.0,
        )
    ]
    write_json(args.out / "manifest.json", cases)
    write_json(args.out / "versions.json", versions())
    launched = 0
    for c in cases:
        path = args.out / c["name"]
        if (path / "summary.json").exists():
            continue
        if args.limit is not None and launched >= args.limit:
            break
        path.mkdir(exist_ok=True)
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
            for c in cases
            if (args.out / c["name"] / "summary.json").exists()
        ],
    )


if __name__ == "__main__":
    main()
