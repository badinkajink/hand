#!/usr/bin/env python3
"""SR2 contact audit: frozen regression, static contact fields, and constraint scaling.

Use the report's pinned interpreter: logs/20261001-hom_contact/venv/bin/python.
Each experiment accepts --dry-run; manifests precede execution. Full per-step
contact logs use gzip CSV; no changes to the existing report/controller scripts.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import time
import traceback
from pathlib import Path

import mujoco
import numpy as np

from contact_surface.geometry import Surface
from contact_surface.metrics import aggregate, contacts
from contact_surface.mujoco_model import build, continuum, load_step, static_fixture
from contact_surface.records import CsvLog, ROOT, versions, write_json

DOC = ROOT / "docs/experiments/20261004-codex"
DEFAULTS = dict(
    backend="mujoco",
    geometry="plane",
    object_radius_mm=12.5,
    spacing_mm=1.0,
    relaxation=0.02,
    phase_degrees=0.0,
    mass_scale=1.0,
    inertia_scale=1.0,
    timestep=0.001,
    iterations=200,
    tolerance=1e-10,
    impratio=10.0,
    seed=0,
    mode="load",
    load=1.0,
    indentation_mm=0.2,
    duration=0.6,
)


def expand(path, phase):
    spec = json.loads(Path(path).read_text())
    configs = []
    if phase == "regression":
        return [dict(phase=phase, **entry) for entry in spec["runs"]]
    for group in spec["groups"]:
        grid = group.get("grid", {})
        for values in itertools.product(*grid.values()):
            c = dict(DEFAULTS, **spec.get("defaults", {}))
            c.update(group.get("fixed", {}))
            c.update(zip(grid, values))
            c.update(phase=phase, group=group["name"])
            # The existing 0.5 mm chain uses 30 ms, because 20 ms is not representable.
            c["relaxation"] = spec.get("relaxation_by_spacing", {}).get(
                str(c["spacing_mm"]), c["relaxation"]
            )
            configs.append(c)
    return configs


def run_id(c):
    digest = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:12]
    return f"{c['phase']}-{c.get('group', c.get('backend', ''))}-{digest}"


def measured(model, data, p, t):
    rows = contacts(model, data, p, t)
    total = aggregate(rows, p)
    return rows, total


def static_run(c, out):
    start = time.perf_counter()
    m, d, p, xml = build(c)
    build_s = time.perf_counter() - start
    (out / "model.xml").write_text(xml)
    write_json(out / "parameters.json", p)
    indentation = c["indentation_mm"] * 1e-3
    final_rows = []
    with CsvLog(out / "contacts.csv.gz") as contact_log, CsvLog(out / "timeseries.csv") as timeline:
        if c["mode"] in ("displacement", "free_snapshot"):
            if c["mode"] == "displacement":
                wrench, acceleration = static_fixture(m, d, p, indentation)
            else:
                d.qpos[2] -= indentation
                mujoco.mj_forward(m, d)
                wrench, acceleration = np.zeros(6), float(np.max(abs(d.qacc)))
            final_rows, total = measured(m, d, p, 0.0)
            for row in final_rows:
                contact_log.row(row)
            timeline.row(dict(time=0.0, indentation=indentation, **total))
            total["forward_equilibrium_acceleration_max"] = acceleration
            total["support_wrench"] = wrench.tolist()
            total["equilibrium"] = acceleration < 1e-5
            total["evaluation"] = (
                "static inverse reaction, checked with native forward solve"
                if c["mode"] == "displacement"
                else "unsupported instantaneous forward solve"
            )
        else:
            # Independent forward simulation from a nearly touching, unloaded state.
            d.qpos[2] -= 0.00001
            history = []
            step_times = []
            for k in range(round(c["duration"] / c["timestep"])):
                t0 = time.perf_counter()
                load_step(m, d, p, c["load"])
                elapsed = time.perf_counter() - t0
                # mj_step's derived/contact data describes the PRE-integration state.
                t = (k) * c["timestep"]
                final_rows, total = measured(m, d, p, t)
                current_indent = p["zero_indentation_height"] - float(d.xpos[1, 2])
                for row in final_rows:
                    contact_log.row(row)
                timeline.row(
                    dict(
                        time=t,
                        indentation=current_indent,
                        Fn_command=c["load"],
                        state="constant_load",
                        step_time_ms=elapsed * 1e3,
                        **total,
                    )
                )
                history.append((total["wrench_2"], current_indent, float(np.linalg.norm(d.qvel))))
                if k >= 50:
                    step_times.append(elapsed)
                if not np.isfinite(d.qpos).all() or abs(d.qpos[2]) > 0.5:
                    raise RuntimeError("Forward integration diverged")
            tail = np.array(history[-max(10, round(0.1 / c["timestep"])) :])
            indentation = float(tail[:, 1].mean())
            total.update(
                equilibrium=bool(
                    np.max(abs(tail[:, 0] - c["load"])) < 0.005 * c["load"]
                    and np.ptp(tail[:, 1]) < 1e-6
                    and tail[-1, 2] < 1e-3
                ),
                force_balance_error=float(tail[:, 0].mean() / c["load"] - 1),
                indentation_tail_range=float(np.ptp(tail[:, 1])),
                speed_final=float(tail[-1, 2]),
                physics_step_ms=float(np.mean(step_times) * 1e3),
                evaluation="native forward dynamics under constant load; five-axis fixture",
            )
        with CsvLog(out / "equilibrium_contacts.csv") as log:
            for row in final_rows:
                log.row(row)
    addresses = [r["constraint_address"] for r in final_rows]
    jac = d.efc_J.reshape(d.nefc, m.nv)
    total["normal_jacobian_rank"] = int(np.linalg.matrix_rank(jac[addresses])) if addresses else 0
    total["constraint_rows"] = int(d.nefc)
    total["object_lateral_offset"] = float(np.linalg.norm(d.xpos[1, :2]))
    total["object_tilt_radians"] = float(2 * np.arccos(np.clip(abs(d.qpos[3]), 0, 1)))
    reference = (
        dict(force=None, arm=None, area=None)
        if c.get("tip_shape") == "filleted_prism"
        else continuum(indentation, c["geometry"], c["object_radius_mm"] * 1e-3, p)
    )
    total.update(
        status="complete",
        indentation=indentation,
        build_s=build_s,
        continuum_force=reference["force"],
        continuum_arm=reference["arm"],
        continuum_area=reference["area"],
        continuum_force_error=total["wrench_2"] / reference["force"] - 1
        if reference["force"]
        else None,
        continuum_arm_error=total["pressure_weighted_arm"] / reference["arm"] - 1
        if total["pressure_weighted_arm"] is not None and reference["arm"]
        else None,
        predicted_mass_stiffness_ratio=1.0
        if p["mapping_policy"] == "lambda_compensated_direct"
        else p["nominal_inverse_inertia"] / p["actual_inverse_inertia"],
        actual_inverse_inertia=p["actual_inverse_inertia"],
        warnings=[
            dict(number=int(w.number), lastinfo=int(w.lastinfo)) for w in d.warning if w.number
        ],
    )
    return total


def regression(c, out):
    import hom_chain as chain

    frozen = json.loads((DOC / "baseline.json").read_text())
    expected = frozen["runs"][c["backend"]]
    original_make = chain.make_plant
    write_json(
        out / "controller.json",
        dict(
            rate=chain.RATE,
            F0=chain.F0,
            F_hold=chain.F_HOLD,
            lift=chain.LIFT,
            brake="closed",
            seed=0,
            trial=expected["trial"],
            source="unchanged hom_chain.run_chain",
            contact_log_decimation=10,
        ),
    )
    with CsvLog(out / "contacts.csv.gz") as contact_log:

        def instrumented_make(*args, **kwargs):
            plant = original_make(*args, **kwargs)
            if plant.sim == "mujoco":
                (out / "model.xml").write_text(plant.xml)
                p = Surface().parameters()
                step = plant.step

                def observed_step(dt):
                    step(dt)
                    for contact in contacts(plant.m, plant.d, p, plant.t - plant.m.opt.timestep):
                        contact_log.row(contact)

                plant.step = observed_step
            else:
                # Drake resultants are already in the unchanged task timeseries;
                # full hydroelastic face logging is deferred to the isolated comparison.
                write_json(
                    out / "contact_logging.json",
                    {
                        "available": False,
                        "reason": "Regression logs aggregate Drake forces; no per-face instrumentation in this pass.",
                    },
                )
            return plant

        try:
            chain.make_plant = instrumented_make
            row = chain.run_chain(expected["spec"], seed=0, brake="closed", perturb=False)
        finally:
            chain.make_plant = original_make
    cols, values = row.pop("trace_cols"), row.pop("trace")
    with CsvLog(out / "timeseries.csv") as log:
        for value in values:
            log.row(dict(zip(cols, value)))
    errors = {k: abs(row[k] - expected[k]) for k in ("phi_end", "end_depth_mm")}
    passed = row["chain_ok"] and errors["phi_end"] <= 0.1 and errors["end_depth_mm"] <= 0.1
    row.update(
        status="complete" if passed else "failed",
        regression_passed=bool(passed),
        reference=expected,
        absolute_errors=errors,
        tolerances={"phi_end": 0.1, "end_depth_mm": 0.1},
    )
    return row


def summarize(out, manifest):
    rows = []
    for item in manifest:
        p = out / item["run_id"] / "summary.json"
        summary = json.loads(p.read_text()) if p.exists() else {"status": "missing"}
        rows.append(dict(run_id=item["run_id"], **item["config"], **summary))
    write_json(out / "aggregate.json", rows)
    # Flat portable aggregate, preserving nested metadata as JSON cells.
    import csv

    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (out / "aggregate.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()}
            )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "phase", choices=["regression", "static", "sensitivity", "fillet", "compensation"]
    )
    ap.add_argument("--config", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", "--list-config", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    config = args.config or DOC / "configs" / f"{args.phase}.json"
    out = args.out or ROOT / "results/20261004-distributed-contact" / args.phase
    configs = expand(config, args.phase)
    manifest = [dict(run_id=run_id(c), config=c) for c in configs]
    if len({r["run_id"] for r in manifest}) != len(manifest):
        raise ValueError("Duplicate run configurations")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        print(f"{len(manifest)} runs", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    old_manifest = out / "manifest.json"
    if old_manifest.exists() and json.loads(old_manifest.read_text()) != manifest:
        raise ValueError("Manifest differs: choose another --out to preserve previous results")
    write_json(old_manifest, manifest)
    meta = versions()
    launched = 0
    for i, item in enumerate(manifest):
        run = out / item["run_id"]
        if (run / "summary.json").exists() and not args.force:
            previous = json.loads((run / "summary.json").read_text())
            if previous["status"] == "complete":
                continue
        if args.limit is not None and launched >= args.limit:
            break
        run.mkdir(exist_ok=True)
        write_json(run / "config.json", item["config"])
        write_json(run / "versions.json", meta)
        try:
            with (
                (run / "stdout.log").open("w") as log,
                contextlib.redirect_stdout(log),
                contextlib.redirect_stderr(log),
            ):
                summary = (
                    regression(item["config"], run)
                    if args.phase == "regression"
                    else static_run(item["config"], run)
                )
        except Exception as exc:
            summary = dict(status="failed", error=repr(exc), traceback=traceback.format_exc())
        write_json(run / "summary.json", summary)
        launched += 1
        if i % 10 == 0 or summary["status"] != "complete" or args.phase == "regression":
            print(f"{i + 1}/{len(manifest)} {item['run_id']} {summary['status']}", flush=True)
    rows = summarize(out, manifest)
    print(
        {
            state: sum(r["status"] == state for r in rows)
            for state in ["complete", "failed", "missing"]
        }
    )
    if any(r["status"] == "failed" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
