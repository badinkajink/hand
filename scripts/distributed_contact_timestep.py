#!/usr/bin/env python3
"""Restartable 50 Hz rolling-task sweep; isolated workers, native videos and live report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from contact_surface.records import ROOT, versions, write_json

DOC = ROOT / "docs/experiments/20261004-codex"
OUT = ROOT / "results/20261004-distributed-contact/task_timestep"
SCRIPT = Path(__file__).resolve()
CURRENT = ROOT / "logs/20261004-contact-current/venv/bin/python"
DRAKE = ROOT / "logs/20261001-hom_contact/venv/bin/python"
POLICIES = (
    ("legacy", "legacy"),
    ("legacy_exact", "legacy_diagexact"),
    ("physical", "physical"),
    ("physical_exact", "physical_diagexact"),
    ("runtime_approx", "physical_runtime_approx"),
    ("runtime_exact", "physical_runtime_diagexact"),
    ("drake", "legacy"),
)


def cases():
    # Nominal/coarse cases first; expensive fine reference remains part of the manifest.
    for dt in (0.001, 0.0005, 0.002, 0.005, 0.01, 0.00005):
        for label, policy in POLICIES:
            yield dict(id=f"roll50_{label}_{round(dt * 1e6)}us", label=label,
                       spec="drake:hydro:rt0.03:hr1" if label == "drake" else
                       "mj:spheres:s1:rs0.75:tr0.03:ir100",
                       contact_policy=policy, physics_dt=dt, controller_hz=50,
                       physics_steps_per_control=round(0.02 / dt), seed=0)


def tool_wrench(pl):
    """Native latest-solve wrench on tool, world axes about current tool origin: F, tau.

    Includes support contacts. Boundary samples, not controller-interval impulse averages.
    MuJoCo's contact arrays describe the last solve before its position integration.
    """
    origin = pl.state()["tool_pos"]
    total = np.zeros(6)
    if pl.sim == "mujoco":
        f = np.zeros(6)
        for i in range(pl.d.ncon):
            c = pl.d.contact[i]
            bodies = pl.m.geom_bodyid[np.asarray(c.geom, int)]
            if pl.tool not in bodies:
                continue
            pl.mj.mj_contactForce(pl.m, pl.d, i, f)
            sign = 1 if bodies[1] == pl.tool else -1
            rotation = np.asarray(c.frame).reshape(3, 3).T
            force, torque = sign * rotation @ f[:3], sign * rotation @ f[3:]
            total[:3] += force
            total[3:] += torque + np.cross(c.pos - origin, force)
    else:
        cr = pl.plant.get_contact_results_output_port().Eval(pl.pc)
        for i in range(cr.num_hydroelastic_contacts()):
            info = cr.hydroelastic_contact_info(i)
            surface = info.contact_surface()
            if not ({surface.id_M(), surface.id_N()} & pl.tool_gids):
                continue
            sign = 1 if surface.id_M() in pl.tool_gids else -1
            force = sign * np.asarray(info.F_Ac_W().translational())
            total[:3] += force
            total[3:] += sign * np.asarray(info.F_Ac_W().rotational())
            total[3:] += np.cross(np.asarray(surface.centroid()) - origin, force)
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            pair = info.point_pair()
            if not ({pair.id_A, pair.id_B} & pl.tool_gids):
                continue
            force = (1 if pair.id_B in pl.tool_gids else -1) * np.asarray(info.contact_force())
            total[:3] += force
            total[3:] += np.cross(np.asarray(info.contact_point()) - origin, force)
    return total


def worker(case):
    import hom_chain as chain
    import hom_paper_tasks as paper

    directory = OUT / case["id"]
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "case.json", case)
    provenance = versions()
    provenance["source_sha256"][str(SCRIPT.relative_to(ROOT))] = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    write_json(directory / "versions.json", provenance)
    film = DOC / "media" / f"timestep-{case['id']}.mp4"
    chain.RATE = case["controller_hz"]
    audits = []

    class AuditedRollout(paper.Rollout):
        time_tolerance = 1e-8

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            audits.append(self)
            self.samples, self.step_times = [], []
            self.title += f" | {case['label']} | control 20 ms | physics {case['physics_dt'] * 1000:g} ms"
            self.csv_file = (directory / "native_state.csv").open("w", newline="")
            self.csv = None
            pl = self.plant
            metadata = dict(backend=pl.sim, info=getattr(pl, "info", {}),
                            sample_semantics="50 Hz boundary state; latest-solve world wrench about tool origin; geometric contact-frame slip proxy")
            (directory / "model.xml").write_text(pl.xml if pl.sim == "mujoco" else pl.xml_mirror)
            if pl.sim == "mujoco":
                metadata.update(engine=pl.mj.__version__, timestep=pl.m.opt.timestep,
                                integrator=int(pl.m.opt.integrator), solver=int(pl.m.opt.solver),
                                iterations=int(pl.m.opt.iterations), tolerance=pl.m.opt.tolerance,
                                enableflags=int(pl.m.opt.enableflags), disableflags=int(pl.m.opt.disableflags))
                tc = pl.info.get("solref_timeconst")
                if tc:
                    refsafe = not bool(pl.m.opt.disableflags & pl.mj.mjtDisableBit.mjDSBL_REFSAFE)
                    metadata.update(refsafe=refsafe, nominal_timeconst_s=tc,
                                    effective_timeconst_s=max(tc, 2 * pl.m.opt.timestep) if refsafe else tc)
            write_json(directory / "model_metadata.json", metadata)
            original_step = pl.step

            def timed_step(duration):
                before = pl.t
                start = time.perf_counter()
                original_step(duration)
                self.step_times.append(time.perf_counter() - start)
                if not np.isclose(pl.t - before, duration, atol=1e-8, rtol=0):
                    raise paper.TaskEscape(self, "physics clock did not advance by the controller interval")

            pl.step = timed_step

        def tick(self, phase, line=None, log=None):
            super().tick(phase, line, log)
            st = self.st
            if (not all(np.isfinite(v).all() for v in st.values())
                    or np.linalg.norm(st["tool_pos"] - self.lay["c"]) > 0.5
                    or np.max(np.abs(st["tool_w"])) > 2000):
                raise paper.TaskEscape(self, "nonfinite/0.5 m escape/2000 rad/s bound")
            if self.plant.sim == "mujoco" and np.any(self.plant.d.warning.number):
                raise paper.TaskEscape(self, "MuJoCo numerical/capacity warning")
            slip = max(float(np.linalg.norm(self.mir.achieved(f, self.fr[f])[0][4:6]))
                       * paper.C.ELL * 1000 for f in ("thumb", "index"))
            wrench = tool_wrench(self.plant)
            row = dict(t=self.t, phase=phase, slip_proxy_mmps=slip,
                       physics_wall_s=self.step_times[-1],
                       command=(log or {}).get("ref", 0), achieved=(log or {}).get("ach", 0))
            for prefix, values in (("pos", st["tool_pos"]), ("quat", st["tool_quat"]),
                                   ("omega", st["tool_w"]), ("wrench", wrench)):
                row.update({f"{prefix}_{i}": float(x) for i, x in enumerate(values)})
            contacts = self.plant.contacts()
            for finger in ("thumb", "index"):
                row[f"N_{finger}"] = contacts[finger]["N"]
                row[f"contacts_{finger}"] = contacts[finger]["n"]
            self.samples.append(row)
            if self.csv is None:
                self.csv = csv.DictWriter(self.csv_file, fieldnames=list(row))
                self.csv.writeheader()
            self.csv.writerow(row)
            self.csv_file.flush()

    paper.Rollout = AuditedRollout
    start = time.perf_counter()
    result = {}
    try:
        result = paper.exp2(case["spec"], film=film, contact_policy=case["contact_policy"],
                            physics_dt=case["physics_dt"])
        write_json(directory / "trace.json", dict(columns=result["trace_cols"], rows=result.pop("trace")))
        write_json(directory / "controller_rows.json", result.pop("rows"))
        result["status"] = "complete"
    except paper.TaskEscape as error:
        result = dict(status="stopped_at_guard", guard_failure=str(error), sim_s=error.rollout.t)
        error.rollout.film_out(film)
    except Exception as error:
        result = dict(status="exception", exception=repr(error), traceback=traceback.format_exc())
        if audits:
            audits[-1].film_out(film)
    finally:
        if audits:
            ro = audits[-1]
            ro.csv_file.close()
            durations = np.asarray(ro.step_times)
            warm = durations[5:]  # exclude first 100 ms from reported quantiles
            result["timing"] = dict(physics_wall_s=float(durations.sum()),
                                    controller_intervals=len(durations),
                                    physics_seconds_per_sim_second=float(durations.sum() / ro.t) if ro.t else None,
                                    p50_interval_ms=float(np.median(warm) * 1000) if len(warm) else None,
                                    p95_interval_ms=float(np.percentile(warm, 95) * 1000) if len(warm) else None,
                                    excludes="model setup, controller, diagnostics, rendering, encoding")
            active = [r for r in ro.samples if r["phase"].startswith("step ")]
            roll = [r for r in active if r["phase"].startswith("step w_tool")]
            result["task_diagnostics"] = dict(
                slip_proxy_integral_mm=sum(r["slip_proxy_mmps"] * ro.dt for r in active),
                tool_roll_positive_deg=sum(r["achieved"] * ro.dt for r in roll if r["command"] > 0),
                tool_roll_negative_deg=sum(r["achieved"] * ro.dt for r in roll if r["command"] < 0),
                pose_and_wrench="native_state.csv",
                retention_success=bool(result.get("pick_ok") and result.get("held_end") and result.get("status") == "complete"))
    result.update(case=case, elapsed_wall_s=time.perf_counter() - start,
                  film=str(film.relative_to(DOC)) if film.exists() else None)
    write_json(directory / "summary.json", result)
    if film.exists():
        check = subprocess.run(["ffmpeg", "-v", "error", "-i", str(film), "-f", "null", "-"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        result["video_decode_ok"] = check.returncode == 0
        result["video_decode_error"] = check.stderr or None
        write_json(directory / "summary.json", result)
    print(case["id"], result["status"], flush=True)


def report():
    from distributed_contact_page import frame, section

    summaries, rows, clips = [], [], []
    for case in cases():
        path = OUT / case["id"] / "summary.json"
        s = json.loads(path.read_text()) if path.exists() else dict(case=case, status="pending")
        summaries.append(s)
        timing = s.get("timing", {})
        task = s.get("per_comp", {}).get("w_tool", {})
        def fmt(x):
            return "—" if x is None else f"{x:.3g}"
        video = f'<a href="{html.escape(s["film"])}">video</a>' if s.get("film") else "—"
        values = [case["label"], f'{case["physics_dt"] * 1000:g}', s["status"],
                  str(s.get("task_diagnostics", {}).get("retention_success", "—")),
                  fmt(task.get("gain")), fmt(task.get("slide_rms_mmps")),
                  fmt(s.get("task_diagnostics", {}).get("slip_proxy_integral_mm")),
                  fmt(timing.get("physics_seconds_per_sim_second")), fmt(timing.get("p95_interval_ms"))]
        rows.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + f"<td>{video}</td></tr>")
        if s.get("film"):
            clips.append(f'<figure><video controls preload="none" playsinline src="{html.escape(s["film"])}" '
                         f'style="width:100%"></video><figcaption>{case["id"]}: {s["status"]}</figcaption></figure>')
    write_json(DOC / "data/task_timestep_results.json", summaries)
    counts = {k: sum(s["status"] == k for s in summaries) for k in sorted({s["status"] for s in summaries})}
    head = ["Variant", "dt (ms)", "Execution", "Picked + retained", "Tool roll gain", "Slide RMS mm/s",
            "Task slip proxy mm", "Physics s / sim s", "p95 physics / control ms", "Native video"]
    body = section("Batch status", f'<p>{html.escape(str(counts))}</p><p>42 nominal cases: the existing '
                   'six-component object-twist task at 50 Hz, seven model variants and six physics steps. '
                   'All variants keep the same 1 mm surface, seed, commands and controller gains. '
                   'MuJoCo uses 3.14; Drake uses the existing 1.57 environment. '
                   'This is a nominal timestep screen, not a robustness sweep or an exact reproduction of the paper.</p>'
                   '<p><a href="data/task_timestep_manifest.json">Frozen manifest</a> · '
                   '<a href="data/task_timestep_results.json">All results, including pending/failures</a> · '
                   '<a href="20261004-contact_research_asides.html#timestep-benchmark">Protocol and motivation</a></p>', "status")
    body += section("Task behavior and cost", '<div class="tw"><table><thead><tr>' +
                    ''.join(f'<th>{h}</th>' for h in head) + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
                    '<p>Execution completion is separate from pickup/retention and tracking quality. '
                    'The slip integral is the maximum of two geometric contact-frame speeds, sampled at 50 Hz; '
                    'it is not a resolved material-point slip distance. Native-state CSVs retain object pose and '
                    'latest-solve wrench in world axes about the tool origin, including support forces. '
                    'These are boundary samples, not interval impulse averages. No claim of high-frequency '
                    'wrench convergence follows from them.</p><p>Physics timing excludes setup, controller, '
                    'logging, rendering and encoding; quantiles discard the first five control intervals. '
                    'End-to-end wall time is also saved. Positive-solref refsafe clamps are logged in model '
                    'metadata; 10 ms physics can soften the legacy contact law. A coarse-step difference '
                    'therefore need not be purely integration error.</p>', "metrics")
    body += section("What the earlier geometry work finished", '<p>The two declared sliding manifests '
                    'were fully executed: force ramp 10/17 completed and seven exited the fixture; guided '
                    '3 mm slide 17/17 completed. All 34 cases remain in the task/sliding report. '
                    'This does not complete the original specification: the static first pass covered '
                    'plane and nominal cylinder, while the sliding extension samples broader cylinders/spheres '
                    'at one load with limited 0.5 mm and phase checks. The complete geometry × load × resolution '
                    '× phase study and matched Drake static geometry comparison remain open. Newton dynamic '
                    'comparison is blocked by unreproduced preload/retention, not assigned a matched task score.</p>'
                    '<p><a href="20261004-task_contact_benchmark.html">Earlier results and videos</a> · '
                    '<a href="20261004-distributed_contact_spec.html">Original specification</a></p>', "prior-work")
    body += section("Native-state videos", '<p>Recorded during each actual run at 25 fps, with simulation '
                    'time and control/physics steps labeled. Guarded trials end at their captured state; '
                    'they are not successful full-length executions.</p>' + ''.join(clips), "videos")
    (DOC / "20261004-controller_timestep.html").write_text(frame(
        "Rolling at 50 Hz: physics timestep sweep", "Live batch results, native videos and explicit failure accounting.",
        body, "4 October 2026 · no per-case controller retuning"))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=[c["id"] for c in cases()])
    parser.add_argument("--case", action="append", choices=[c["id"] for c in cases()])
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(next(c for c in cases() if c["id"] == args.worker))
        return
    manifest = dict(controller_period_s=0.02, cases=list(cases()),
                    scope="nominal screen; perturbations and higher-rate wrench capture deferred",
                    schedule_time_tolerance_s=1e-8,
                    provisional_reference_tolerances=dict(roll_lobe_degrees=3.0, gain_absolute=0.05,
                                                          slip_proxy_integral_mm=0.5,
                                                          note="versus same backend/policy at 50 us; require pickup/retention too; screening targets, not hardware validation"),
                    success="report pickup + final retention separately from gain/slip/pose errors; no fidelity pass inferred",
                    timeout_s_per_case=1800, source_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("Manifest/source changed: use a new output directory; do not overwrite existing runs")
    write_json(manifest_path, manifest)
    write_json(DOC / "data/task_timestep_manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        report()
        return
    if args.report_only:
        print(report())
        return
    for case in cases():
        if args.case and case["id"] not in args.case:
            continue
        directory = OUT / case["id"]
        if (directory / "summary.json").exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "case.json", case)
        write_json(OUT / "progress.json", dict(pid=os.getpid(), current=case["id"], state="running"))
        env = dict(os.environ, MUJOCO_GL="egl", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (directory / "stdout.log").open("w") as log:
            try:
                child = subprocess.run([str(DRAKE if case["label"] == "drake" else CURRENT), str(SCRIPT),
                                        "--worker", case["id"]], cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=manifest["timeout_s_per_case"])
                failure = f"worker exit {child.returncode}"
            except subprocess.TimeoutExpired:
                failure = "worker timeout after 1800 s"
            if not (directory / "summary.json").exists():
                write_json(directory / "summary.json", dict(case=case, status="worker_failed", exception=failure))
        print(case["id"], report(), flush=True)
    write_json(OUT / "progress.json", dict(pid=os.getpid(), state="idle", counts=report()))


if __name__ == "__main__":
    main()
