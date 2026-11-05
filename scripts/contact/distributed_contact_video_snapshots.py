#!/usr/bin/env python3
"""Record explicitly static Newton pressure and 6 mm fillet snapshots for movies."""

import argparse
import csv
import gzip
import json

import mujoco
import numpy as np

from contact_surface.records import ROOT, write_json, versions

OUT = ROOT / "results/20261004-distributed-contact/video_capture"
RESULTS = ROOT / "results/20261004-distributed-contact"


def pressure():
    import warp as wp
    import newton
    from distributed_contact_newton import setup
    from contact_surface.video import StateCapture

    cases = []
    write_json(OUT / "versions-pressure.json", versions())
    for h in (1.0, 0.5, 0.25):
        for reduced, matching in [(False, False), (True, False), (True, True)]:
            rid = f"pressure_h{h}_reduced{reduced}_moment{matching}"
            path = OUT / rid
            path.mkdir(exist_ok=True)
            cfg = dict(
                pressure_only=True,
                static_squeeze_m=0.0002,
                voxel_mm=h,
                timestep=0.00005,
                reduce_contacts=reduced,
                moment_matching=matching,
            )
            case = dict(
                backend="pressure",
                phase="newton_pressure",
                run_id=rid,
                video_id=rid,
                config=cfg,
                reference="results/20261004-distributed-contact/newton_pressure",
                title=f"STATIC pressure: {h:g} mm SDF, {'unreduced' if not reduced else 'reduced + moment matching' if matching else 'reduced'}",
            )
            cases.append(case)
            if (path / "summary.json").exists():
                continue
            cap = StateCapture(path, "newton", interval=0.001)
            try:
                for i, delta in enumerate((0.0001, 0.0002, 0.0004)):
                    model, solver, pipe, state, _, _, cc, meta = setup(
                        dict(cfg, static_squeeze_m=delta), path
                    )
                    n = int(cc.rigid_contact_count.numpy()[0])
                    dist = wp.zeros(cc.rigid_contact_max, dtype=float)
                    pos = wp.zeros(cc.rigid_contact_max, dtype=wp.vec3)
                    newton.eval_rigid_contact_kinematics(
                        model, state, cc, out_distance=dist, out_point0_world=pos
                    )
                    fn = cc.rigid_contact_stiffness.numpy()[:n] * np.maximum(0, -dist.numpy()[:n])
                    pts = pos.numpy()[:n]
                    normals = cc.rigid_contact_normal.numpy()[:n]
                    meta["newton_tool"] = next(
                        j
                        for j, label in enumerate(model.body_label)
                        if label.split("/")[-1] == "tool"
                    )
                    points = [
                        [*p.tolist(), *nrm.tolist(), float(f)]
                        for p, nrm, f in zip(pts, normals, fn)
                    ]
                    record = dict(
                        time=i * 0.3,
                        stage="static prescribed snapshot",
                        indentation_mm=delta * 1000,
                        normal_total=float(fn.sum()),
                        contacts=n,
                    )
                    cap.newton(model, None, state, meta, points, record)
                write_json(
                    path / "summary.json",
                    dict(
                        status="complete",
                        config=cfg,
                        record_type="prescribed static snapshots; no integration",
                        snapshots=3,
                        simulated_s=0.0,
                    ),
                )
                write_json(path / "case.json", case)
            finally:
                cap.close()
            print(rid, flush=True)
    return cases


def fillet():
    selected = []
    for p in (RESULTS / "fillet").glob("*/config.json"):
        c = json.loads(p.read_text())
        if (
            c["spacing_mm"] == 0.5
            and c["tilt_degrees"] in [0, 45, 75, 85]
            and c["azimuth_degrees"] == 0
            and c["indentation_mm"] == 0.2
        ):
            selected.append((c, p.parent))
    selected.sort(key=lambda x: x[0]["tilt_degrees"])
    cfg, path0 = selected[0]
    rid = "fillet_6mm_static_tilt"
    path = OUT / rid
    path.mkdir(exist_ok=True)
    case = dict(
        backend="fillet",
        phase="fillet",
        run_id=rid,
        video_id=rid,
        config=cfg,
        reference=str(path0.relative_to(ROOT)),
        title="STATIC 6 mm fillet: front-surface tilt snapshots",
    )
    states = []
    for i, (c, source) in enumerate(selected):
        m = mujoco.MjModel.from_xml_path(str(source / "model.xml"))
        p = json.loads((source / "parameters.json").read_text())
        q = m.qpos0.copy()
        q[:3] -= 0.0002 * np.array(p["contact_axis"])
        with (source / "equilibrium_contacts.csv").open() as f:
            rows = list(csv.DictReader(f))
        points = [
            [
                *[float(r[f"position_{j}"]) for j in range(3)],
                *[float(r[f"normal_{j}"]) for j in range(3)],
                float(r["f_n"]),
                float(r["pressure_proxy"]),
                float(r["A_i"]),
                int(r["geom_sample"]),
            ]
            for r in rows
        ]
        states.append(
            dict(
                time=i * 0.3,
                qpos=q.tolist(),
                contacts=points,
                metrics=dict(
                    stage="static prescribed snapshot",
                    tilt_degrees=c["tilt_degrees"],
                    normal_total=sum(p[6] for p in points),
                ),
            )
        )
    with gzip.open(path / "states.jsonl.gz", "wt") as f:
        for s in states:
            f.write(json.dumps(s, allow_nan=False) + "\n")
    write_json(
        path / "summary.json",
        dict(
            status="complete",
            config=cfg,
            record_type="original native static equilibria; no integration",
            snapshots=4,
            simulated_s=0.0,
        ),
    )
    write_json(path / "case.json", case)
    print(rid, flush=True)
    return [case]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["pressure", "fillet"])
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    cases = pressure() if a.kind == "pressure" else fillet()
    write_json(OUT / f"snapshots-{a.kind}.json", cases)


if __name__ == "__main__":
    main()
