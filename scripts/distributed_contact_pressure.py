#!/usr/bin/env python3
"""Separate Newton hydroelastic pressure quadrature from time integration."""

import numpy as np
import warp as wp
import newton
import hom_contact_rig as H
from contact_surface.records import ROOT, write_json, versions
from distributed_contact_newton import setup

out = ROOT / "results/20261004-distributed-contact/newton_pressure"
out.mkdir(parents=True, exist_ok=True)
write_json(out / "versions.json", versions())
rows = []
for h in (1.0, 0.5, 0.25):
    for delta in (0.0001, 0.0002, 0.0004):
        for reduced, matching in ((False, False), (True, False), (True, True)):
            rid = f"h{h}_d{delta}_red{reduced}_moment{matching}"
            path = out / rid
            path.mkdir(exist_ok=True)
            c = dict(
                pressure_only=True,
                static_squeeze_m=delta,
                voxel_mm=h,
                timestep=0.00005,
                reduce_contacts=reduced,
                moment_matching=matching,
            )
            m, _, pipe, state, _, _, cc, meta = setup(c, path)
            count = int(cc.rigid_contact_count.numpy()[0])
            distances = wp.zeros(cc.rigid_contact_max, dtype=float)
            points = wp.zeros(cc.rigid_contact_max, dtype=wp.vec3)
            newton.eval_rigid_contact_kinematics(
                m, state, cc, out_distance=distances, out_point0_world=points
            )
            dist = distances.numpy()[:count]
            k = cc.rigid_contact_stiffness.numpy()[:count]
            fn = k * np.maximum(0.0, -dist)
            normals = cc.rigid_contact_normal.numpy()[:count]
            pts = points.numpy()[:count]
            arm = np.cross(np.array(meta["u"]), pts - meta["P"])
            vt = arm - (arm * normals).sum(axis=1)[:, None] * normals
            tau = float(np.sum(fn * np.linalg.norm(vt, axis=1)))
            fr = cc.rigid_contact_friction.numpy()[:count]
            fr = np.where(fr > 0.0, fr, 1.0)
            tau_scaled = float(np.sum(fn * fr * np.linalg.norm(vt, axis=1)))
            signs = np.abs(normals @ meta["u"])
            oracle = H.winkler_sphere_cylinder(delta, 1e7, n=401)
            r = dict(
                config=c,
                contact_count=count,
                normal_force_projection=float(np.sum(fn * signs)),
                pressure_force_sum=float(sum(fn)),
                torsion_capacity=tau,
                torsion_capacity_friction_scaled=tau_scaled,
                continuum_normal_2pads=2 * oracle["Fn"],
                continuum_torque_2pads=2 * oracle["Fn"] * oracle["rbar"],
                stiffness_max=float(max(k, default=0.0)),
                minimum_distance_m=float(min(dist, default=0.0)),
                geometry_parity=meta["initial_pose_errors_m"],
            )
            rows.append(r)
            write_json(path / "summary.json", r)
            print(rid, count, r["normal_force_projection"], tau, flush=True)
write_json(out / "summary.json", rows)
