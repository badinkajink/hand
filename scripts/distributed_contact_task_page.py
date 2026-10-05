#!/usr/bin/env python3
"""Build the task and foundational sliding report from native run summaries."""

import csv
import html
import json
import shutil
import subprocess

from contact_surface.records import ROOT, write_json
from distributed_contact_page import frame, section

DOC = ROOT / "docs/experiments/20261004-codex"
RESULTS = ROOT / "results/20261004-distributed-contact"
MEDIA = DOC / "media"


def h(value):
    return html.escape(str(value))


def fmt(value, decimals=2):
    return "—" if value is None else f"{value:.{decimals}f}"


def table(headers, rows):
    return (
        '<div class="table-scroll"><table><thead><tr>'
        + "".join(f"<th>{h(x)}</th>" for x in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def case_summary(name):
    path = RESULTS / "task_chain" / name / "summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def film_card(name, title):
    path = MEDIA / f"task-{name}.mp4"
    if not path.exists():
        return ""
    poster = path.with_suffix(".jpg")
    poster_attr = f' poster="media/{h(poster.name)}"' if poster.exists() else ""
    return (f'<figure><video controls preload="metadata" playsinline{poster_attr} '
            f'src="media/{h(path.name)}"></video><figcaption>{h(title)} · '
            f'<a href="media/{h(path.name)}">MP4</a></figcaption></figure>')


def film_grid(name, ids, prefix="task-", crop_width=480, crop_height=360):
    paths = [MEDIA / f"{prefix}{x}.mp4" for x in ids]
    if any(not p.exists() for p in paths):
        return None
    output = MEDIA / f"compare-task-{name}.mp4"
    poster = output.with_suffix(".jpg")
    if output.exists():
        return output
    durations = [float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(p)], text=True)) for p in paths]
    graph = []
    for n in range(4):
        graph.append(f"[{n}:v]crop={crop_width}:{crop_height}:0:0,"
                     "tpad=stop_mode=clone:stop_duration=30,"
                     f"setpts=PTS-STARTPTS[v{n}]")
    graph.append("[v0][v1][v2][v3]xstack=inputs=4:layout="
                 f"0_0|{crop_width}_0|0_{crop_height}|{crop_width}_{crop_height}[v]")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path in paths:
        command += ["-i", str(path)]
    command += ["-filter_complex", ";".join(graph), "-map", "[v]", "-t", str(max(durations)),
                "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "22",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    subprocess.run(command, check=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "0.1",
                    "-i", str(output), "-frames:v", "1", str(poster)], check=True)
    return output


def task_rows(ids, rolling=False):
    rows = []
    for name in ids:
        s = case_summary(name)
        if s is None:
            continue
        video = f'<a href="media/task-{h(name)}.mp4">video</a>' if s.get("film") and (MEDIA / f"task-{name}.mp4").exists() else "—"
        cost = s.get("perf", {}).get("phys")
        if isinstance(cost, dict):
            cost = sum(cost.values())
        rate = cost / s["sim_s"] if cost is not None and s.get("sim_s") else None
        label = h(name.replace("_s1_", " / 1 mm / ").replace("_s05_", " / 0.5 mm / "))
        if rolling:
            gain = s.get("per_comp", {}).get("w_tool", {}).get("gain")
            slide = s.get("per_comp", {}).get("w_tool", {}).get("slide_rms_mmps")
            rows.append([label, h(s["status"]), h(s.get("pick_ok", "—")),
                         h(s.get("held_end", "—")), fmt(gain, 3),
                         fmt(slide, 3), fmt(s.get("rmse_ang_dps"), 1),
                         fmt(rate, 2), video])
        else:
            rows.append([label, h(s["status"]), h(s.get("pick_ok", "—")),
                         h(s.get("chain_ok", "—")), fmt(s.get("lift_axis_tilt_deg"), 1),
                         fmt(s.get("phi_end"), 1), fmt(s.get("slip_mm"), 1),
                         fmt(rate, 2), video])
    headers = (["Case", "Run status", "Picked", "Held at end", "Tool-axis twist gain",
                "Tool-axis slide RMS mm/s", "Angular RMSE °/s", "Physics wall / sim", "Film"] if rolling else
               ["Case", "Run status", "Picked", "Chain pass", "Tilt after lift °",
                "Brake angle °", "Axial slip mm", "Physics wall / sim", "Film"])
    return table(headers, rows)


def geometry_rows(phase):
    base = RESULTS / phase
    if not (base / "manifest.json").exists():
        return "<p>Run pending.</p>", []
    manifest = json.loads((base / "manifest.json").read_text())
    rows, export = [], []
    for entry in manifest:
        c = entry["config"]
        path = base / entry["run_id"]
        s = json.loads((path / "summary.json").read_text())
        last_time = None
        last_z = None
        peak_slip = None
        if (path / "timeseries.csv").exists():
            with (path / "timeseries.csv").open(newline="") as source:
                for record in csv.DictReader(source):
                    last_time, last_z = float(record["time"]), float(record["z"])
                    if last_time >= 0.35:
                        peak_slip = max(peak_slip or 0.0, float(record["slip_speed_max"]) * 1000)
        export.append(dict(run_id=entry["run_id"], config=c, summary=s,
                           last_time_s=last_time, last_z_m=last_z,
                           peak_contact_slip_mmps=peak_slip))
        rows.append([
            h(c["geometry"]), fmt(c["object_radius_mm"], 2) if c["geometry"] != "plane" else "—",
            h(c["slide_axis"]), fmt(c["spacing_mm"], 1), fmt(c["phase_degrees"], 0),
            h(s["status"]), fmt(s.get("slip_distance_mm"), 2), fmt(peak_slip, 1),
            fmt(s.get("normal_law_L1_error"), 3), fmt(s.get("max_active_contacts"), 0),
            fmt(last_time if s["status"] != "complete" else None, 3),
        ])
    name = "task_" + phase + "_results.json"
    write_json(DOC / "data" / name, export)
    return table(["Surface", "Radius mm", "Slide axis", "Spacing mm", "Phase °", "Status",
                  "Travel mm", "Peak local slip mm/s", "Normal-law L1", "Max active", "Exit time s"], rows), export


def main():
    for source, target in (
        (RESULTS / "task_chain/versions.json", "task_chain_versions_36.json"),
        (RESULTS / "task_chain/versions-v314.json", "task_chain_versions_314.json"),
        (RESULTS / "sliding_geometry/manifest.json", "task_sliding_geometry_manifest.json"),
        (RESULTS / "sliding_geometry_guided/manifest.json",
         "task_sliding_geometry_guided_manifest.json"),
    ):
        if source.exists():
            shutil.copy2(source, DOC / "data" / target)
    chain_ids = ["mj_physical_s1_50us", "mj_physical_s05_50us", "mj_physical_s1_500us",
                 "mj_physical_s1_2ms", "mj_legacy_s1_1ms", "mj_point4s_1ms",
                 "drake_hydro_1ms", "drake_hydro_500us", "mj_physical_s1_50us_seed1",
                 "drake_hydro_1ms_seed1"]
    roll_ids = ["roll_mj_physical_s1_50us_ctrl500",
                "roll_mj_physical_s05_50us_ctrl500", "roll_mj_legacy_s1_1ms_ctrl500",
                "roll_drake_hydro_1ms_ctrl500"]
    ablation_ids = [f"v314_{task}_{policy}_s1_50us"
                    for task in ("chain", "roll") for policy in
                    ("legacy", "legacy_exact", "physical", "physical_exact",
                     "physical_runtime_approx", "physical_runtime_exact")]
    parts = []
    parts.append(section("What these runs say", """
    <p><strong>Task behavior does not improve monotonically with a more literal local spring.</strong>
    The precomputed physical-stiffness mapping recovered the desired per-sample static
    coefficient in the earlier fixture, but its 1 mm articulated hand tilts during pickup
    and has weak screwdriver-axis rolling. The legacy MuJoCo spheres and Drake hydroelastic
    contact complete the same chain; the 0.5 mm mapped spheres recover much of the rolling
    response at greater CPU cost. These are native trajectories, with guards and failures retained.</p>
    <p>The <a href="https://arxiv.org/html/2609.25619v2">Wang–Oh–Pollard rolling-pinch
    experiment</a> motivates the second task. We adapt its cylinder-as-effector step-body-twist
    protocol to the existing SR2 scene. It is a matched <em>within-repository</em> comparison,
    not a reproduction of the paper's hardware scores or exact model. Control is 500 Hz in the
    headline rolling table; physics is 1 kHz for Drake and legacy MuJoCo, 20 kHz for mapped MuJoCo.
    Earlier 100 Hz variants are retained in the machine-readable data.</p>
    <p>The legacy sphere model is itself calibrated at the initial pose using the
    approximate body inverse inertia. The mapped variant instead uses direct-format
    coefficients for the target sample stiffness and damping, constant impedance
    d₀ = 0.0001, <code>impratio</code> = 10000 and a declared tangential damping ratio.
    A legacy-versus-mapped task difference therefore bundles the normal law and contact
    regularization. The within-family diagonal and tangential controls below isolate
    parts of that bundle.</p>
    <p><a href="20261004-contact_videos.html">Earlier contact video gallery</a> ·
    <a href="20261004-articulated_contact_transfer.html">Holding and torsion</a> ·
    <a href="20261004-distributed_contact_spec.html">Original contact specification</a></p>
    """, "outcome"))
    chain_grid = film_grid("chain", ["mj_physical_s1_50us", "mj_physical_s05_50us",
                                    "mj_legacy_s1_1ms", "drake_hydro_1ms"])
    roll_grid = film_grid("rolling", roll_ids[:3] + [roll_ids[3]])
    geometry_manifest = DOC / "data/task_geometry_video_manifest.json"
    geometry_films = json.loads(geometry_manifest.read_text()) if geometry_manifest.exists() else []
    geometry_grid = film_grid("geometry", [row["video_id"] for row in geometry_films[:4]],
                              prefix="", crop_width=1000, crop_height=496) \
        if len(geometry_films) >= 4 else None
    video_body = ""
    for grid, title, caption in (
        (chain_grid, "Pickup, closed brake, insertion",
         "Top: mapped 1 mm / mapped 0.5 mm. Bottom: legacy 1 mm / Drake. A stopped panel freezes; the full clips are below."),
        (roll_grid, "Cylinder-as-effector rolling commands",
         "Top: mapped 1 mm / mapped 0.5 mm. Bottom: legacy 1 mm / Drake. All use the same 500 Hz controller."),
        (geometry_grid, "Guided sliding on distinct geometries",
         "Plane, cylinder along axis, cylinder across curvature and sphere. All request the same 3 mm tangent travel."),
    ):
        if grid:
            video_body += (f'<figure><h3>{h(title)}</h3><video controls preload="metadata" '
                           f'playsinline poster="media/{h(grid.with_suffix(".jpg").name)}" '
                           f'src="media/{h(grid.name)}"></video><figcaption>{h(caption)} '
                           f'<a href="media/{h(grid.name)}">MP4</a></figcaption></figure>')
    video_body += '<div class="video-cards">' + "".join(film_card(name, name) for name in
        ("mj_physical_s1_50us", "mj_physical_s05_50us", "mj_legacy_s1_1ms", "drake_hydro_1ms",
         *roll_ids, "v314_chain_legacy_exact_s1_50us", "v314_chain_physical_exact_s1_50us",
         "v314_chain_physical_runtime_exact_s1_50us", "v314_roll_legacy_exact_s1_50us",
         "v314_roll_physical_exact_s1_50us", "v314_roll_physical_runtime_exact_s1_50us",
         "v314_chain_physical_runtime_ct100_exact_s1_50us",
         "v314_roll_physical_runtime_ct100_exact_s1_50us")) + "</div>"
    video_body += ('<p>The two force-ramp geometry clips stop at the same fixture-exit '
                   'time as their original measured runs; their red badges mark the stop.</p>')
    video_body += '<div class="video-cards">' + "".join(
        f'<figure><video controls preload="metadata" playsinline src="{h(row["src"])}"></video>'
        f'<figcaption>{h(row["title"])} · <a href="{h(row["src"])}">MP4</a></figcaption></figure>'
        for row in geometry_films) + "</div>"
    parts.append(section("Watch the native trajectories", video_body, "films"))
    parts.append(section("The complete hand chain", """
    <p>One fixed-palm SR2 picks up a 24.54 g screwdriver, lifts it, brakes it toward 90°,
    holds, transports, aims and inserts. The same controller, tool and friction are used
    across cases. A pass requires all stages; a run that merely completes is not a pass.
    The two seeds are separate deterministic perturbations, not a statistical success rate.</p>
    """ + task_rows(chain_ids), "chain"))
    parts.append(section("Rolling-pinch transfer at 500 Hz", """
    <p>After pickup, the controller requests positive and negative 20 mm/s translations
    and 40°/s body rotations while suppressing relative contact motion. Tool-axis twist
    gain is achieved angular velocity divided by commanded angular velocity for that step.
    A picked and retained cylinder can still track that twist badly. Lower RMSE alone does
    not establish faithful rolling if cross-axis motion is large.</p>
    """ + task_rows(roll_ids, rolling=True), "rolling"))
    ablation_ids += ["v314_chain_physical_runtime_ct100_exact_s1_50us",
                     "v314_roll_physical_runtime_ct100_exact_s1_50us",
                     "v314_roll_physical_runtime_ct1000_exact_s1_50us"]
    parts.append(section("What diagexact changes", """
    <p>MuJoCo 3.14 ablates the diagonal flag on the same 1 mm hand at 50 µs. The legacy
    pair compares flag off/on without altering its contact parameters. The physical pairs
    compare compiled fixed coefficients and per-step retranslation against the diagonal
    selected by MuJoCo, again with the flag off/on. This tests whether the frozen-body
    approximation caused the task failure; it does not equate a better task score with a
    more faithful material model. In particular, a higher tool-axis twist gain accompanied
    by more relative sliding is not successful rolling contact. Physics cost excludes
    video rendering. The runtime-approximate and fixed-approximate variants agree closely,
    a control for the split-step adapter itself. The runtime-exact variant changes both
    the selected inertia and the contact coefficients to preserve the prescribed local
    stiffness; the fixed-exact variant changes only the selected inertia. The extra
    Cₜ/Cᵢ = 100/1000 rows change tangential damping while holding normal mapping fixed.
    Raising that ratio from 10 to 100 raises tool-axis twist gain from 0.19 to 0.78,
    but slide RMS rises from 4.1 to 16.2 mm/s and angular RMSE from 32.7 to
    56.2°/s. At 1000 the slide reaches 24.8 mm/s. A higher twist gain here is
    more motion through the contact, not recovered rolling.</p>
    """ + task_rows(ablation_ids[:6] + ablation_ids[12:13])
    + task_rows(ablation_ids[6:12] + ablation_ids[13:], rolling=True), "diagexact"))
    force_table, force = geometry_rows("sliding_geometry")
    guided_table, guided = geometry_rows("sliding_geometry_guided")
    parts.append(section("Fundamental sliding geometry", """
    <p>The initial force-ramp sweep revisits the specification's plane, cylinders of radius
    6.25/12.5/25/50 mm and spheres of radius 12.5/25 mm. Cylinder <code>x</code> is axial;
    <code>y</code> crosses its curvature. Samples use 1 mm spacing, with selected 0.5 mm and
    30°/60° phase controls. The same 1 N normal load, 1.2 N tangential ramp, braking
    schedule and 50 µs step are used in each run. A fixture exit is a real loss of
    controlled contact, not an unreported solver result. The table's exit time is the
    last saved physics step when a run fails. In this setup, 6.25 and 12.5 mm cylinders
    complete axial travel but cross-curvature motion leaves the local patch around
    0.76–0.78 s. Increasing sample resolution or rotating its phase does not prevent
    that loss. The larger 25 and 50 mm cylinders remain in the fixture.</p>
    """ + force_table + """
    <p>The guided-path follow-up requests only 3 mm of travel with the same 1 N load and a
    declared 1000 N/m normal height spring that follows each analytic surface profile.
    This holds the comparison in the local contact patch. It changes the fixture, so its
    force and travel scores should not be compared numerically with the force-ramp runs.
    Both protocols use the runtime-selected local stiffness mapping and diagexact.
    At 12.5 mm cylinder radius, the 1 and 0.5 mm across-curvature cases both travel
    about 2.40 mm with about 9.8 mm/s peak local slip, while maximum active samples
    rise from 11 to 43. The 30°/60° phase controls are similar. The 50 mm cases
    remain below the 1 mm/s local-slip threshold, so their travel is a sticking
    control rather than a sliding comparison.</p>
    """ + guided_table, "geometry"))
    parts.append(section("Reproduce and limits", """
    <p>Raw per-run configurations, MJCF, aggregate traces, contact logs and solver-coupling
    snapshots live in <code>results/20261004-distributed-contact/</code>. Portable summaries
    and video manifests are in <a href="data/task_chain_results.json">task data</a>,
    <a href="data/task_diagexact_results.json">diagexact data</a>,
    <a href="data/task_sliding_geometry_results.json">force-ramp geometry</a> and
    <a href="data/task_sliding_geometry_guided_results.json">guided geometry</a>.
    Source/package provenance is in the <a href="data/task_chain_versions_36.json">pinned</a>
    and <a href="data/task_chain_versions_314.json">current-engine</a> version records.</p>
    <pre>MUJOCO_GL=egl logs/20261001-hom_contact/venv/bin/python scripts/distributed_contact_tasks.py
MUJOCO_GL=egl logs/20261004-contact-current/venv/bin/python scripts/distributed_contact_diagexact_tasks.py
logs/20261004-contact-current/venv/bin/python scripts/distributed_contact_dynamics.py slip --config docs/experiments/20261004-codex/configs/sliding_geometry.json --out results/20261004-distributed-contact/sliding_geometry
logs/20261004-contact-current/venv/bin/python scripts/distributed_contact_dynamics.py slip --config docs/experiments/20261004-codex/configs/sliding_geometry_guided.json --out results/20261004-distributed-contact/sliding_geometry_guided
MUJOCO_GL=egl logs/20261001-hom_contact/venv/bin/python scripts/distributed_contact_task_page.py</pre>
    <p>Newton's earlier native pressure/reduction transfer runs are a readiness gate, not a
    matched pickup or rolling score. The articulated task has not been ported to Newton;
    the <a href="20261004-articulated_contact_transfer.html">backend report</a> records
    the pressure/contact instability and the transfer limitations. The 6 mm hardware
    fillet has a static audit but no dynamic sliding task here. The spherical fingertips
    remain the task baseline.</p>
    <figure><video controls preload="metadata" playsinline
    src="media/newton_transfer-native_h05.mp4"></video>
    <figcaption>Newton native hydroelastic preload gate: stopped at 0.075 s before a
    task-level controller can be scored. <a href="media/newton_transfer-native_h05.mp4">MP4</a> ·
    <a href="20261004-contact_videos.html#newton-replay">Replay audit</a>.</figcaption></figure>
    """, "reproduce"))
    toc = ('<nav class="toc"><a href="#outcome">Finding</a><a href="#films">Videos</a>'
           '<a href="#chain">Chain</a><a href="#rolling">Rolling</a>'
           '<a href="#diagexact">diagexact</a><a href="#geometry">Geometry</a>'
           '<a href="#reproduce">Reproduce</a></nav>')
    page = frame("Contact models on robot tasks and sliding geometry",
                 "Native SR2 pickup, braking, insertion and rolling-pinch videos, with a direct "
                 "MuJoCo diagonal ablation and curvature-sensitive sliding fixtures.",
                 toc + "".join(parts),
                 "Native simulations · fixed controller and declared contact variants")
    page = page.replace("</style>", """
    .video-cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}
    video{width:100%;height:auto;background:#10151a}
    figure{margin:1.2rem 0} figure h3{margin-bottom:.4rem}
    .table-scroll{overflow-x:auto;margin:1rem 0} table{min-width:750px}
    @media(max-width:760px){.video-cards{grid-template-columns:1fr}}
    </style>""")
    output = DOC / "20261004-task_contact_benchmark.html"
    output.write_text(page)
    print(output, len(force), len(guided))


if __name__ == "__main__":
    main()
