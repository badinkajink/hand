#!/usr/bin/env python3
"""Synchronized video comparisons and a gallery built from real state captures."""

import html
import json
import subprocess

from contact_surface.records import ROOT, write_json
from distributed_contact_page import frame, section
import retro_style

DOC = ROOT / "docs/experiments/20261004-codex"
MEDIA = DOC / "media"


def comparisons(films):
    byid = {f["video_id"]: f for f in films}
    groups = [
        (
            "resolution",
            "Sphere resolution: the same 12 mN·m pulse",
            [
                "transfer-fine_s1.0_tau0.012",
                "transfer-fine_s0.5_tau0.012",
                "transfer-refine_s0.25",
                "drake_transfer-drake_tau0.012",
            ],
            "1 mm / 50 µs, 0.5 mm / 50 µs, 0.25 mm / 25 µs and Drake / 500 µs. Total sample area and stiffness stay fixed. All cameras and input pulse timing agree.",
        ),
        (
            "backends",
            "Matched pulse across CPU, GPU and Drake",
            [
                "transfer-fine_s1.0_tau0.012",
                "mjlab_transfer-fine_s1.0_batch128",
                "transfer-fine_s0.5_tau0.012",
                "drake_transfer-drake_tau0.012",
            ],
            "GPU footage replays actual world 0 of the 128-world batch. Its neutral sample colors reflect missing per-point force logs. The other cases show native contact pressure in their individual close-ups.",
        ),
        (
            "creep",
            "Small-torque holding and friction creep",
            [
                "transfer-fine_s0.5_tau0.003",
                "friction_transfer-creep_ct100.0",
                "friction_refine-creep_ct1000_dt6.25",
                "drake_transfer-drake_tau0.003",
            ],
            "The same 3 mN·m pulses with Cₜ/Cᵢ = 10, 100 and 1000, then Drake. The 1000 case uses 6.25 µs; the other CPU cases use 50 µs. These are declared operating points, not a solver-only comparison.",
        ),
        (
            "holding",
            "Holding ramps across contact resolution",
            [
                "holding-hold_s2.0_m1.0",
                "holding-hold_s1.0_m1.0",
                "holding-hold_s0.5_m1.0",
                "holding-hold_load4_s0.5",
            ],
            "2, 1 and 0.5 mm spacing at nominal preload; the fourth panel doubles requested pad load. The fourth ramp is faster. Motion thresholds and measured normal load are exported alongside the videos.",
        ),
        (
            "mass",
            "Mass and timestep sensitivity",
            [
                "holding-hold_s0.5_m0.25",
                "holding_refine-quarter_s0.5_dt1.25e-05",
                "holding-hold_s0.5_m1.0",
                "holding-hold_s0.5_m4.0",
            ],
            "Quarter mass at 25 µs and 12.5 µs, nominal mass at 50 µs, then four-times mass at 50 µs. The refined quarter-mass ramp is slower and shorter. 200 Hz video sampling cannot resolve every physics-step load oscillation; the underlying traces are retained.",
        ),
        (
            "payload",
            "Payload retention at matched resolution",
            [
                "payload-payload_s1_m8",
                "payload-payload_s05_m8",
                "payload-payload_s1_m16",
                "payload-payload_s05_m16",
            ],
            "196.3 g at 1 and 0.5 mm, then 392.7 g at 1 and 0.5 mm. Object-only weight ramps after settling; no translational spring supports it. Red badges identify the 25 mm escape guard. Retention with drift is visible.",
        ),
        (
            "release",
            "Strong-torque release across resolutions",
            [
                "transfer-fine_s1.0_tau0.048",
                "transfer-fine_s0.5_tau0.048",
                "transfer-fast_refine_s025",
                "drake_transfer-drake_tau0.048",
            ],
            "The same 48 mN·m pulses. Each stopped panel freezes its last recorded state with a red badge; it is not continued motion. Drake's differing trajectory is retained.",
        ),
        (
            "numerics",
            "Timestep and impedance controls",
            [
                "transfer-compiled_s0.5",
                "transfer-fine_s0.5_tau0.012",
                "friction_transfer-creep_ct1000.0",
                "transfer_compiled_fine-compiled_fine_s05",
            ],
            "Original impedance / 500 µs, fine normal law / 50 µs, underresolved strong-friction damping / 50 µs, and the precomputed fine mapping / 50 µs. The third panel uses the smaller 3 mN·m pulse; its oscillation must not be read as a matched-input improvement.",
        ),
        (
            "isolated",
            "Isolated normal pulses and controlled sliding",
            [
                "isolated-plane-normal_pulse",
                "isolated-cylinder-normal_pulse",
                "isolated-plane-controlled_slip",
                "isolated-cylinder-controlled_slip",
            ],
            "Actual native isolated-fixture trajectories at 0.5 mm spacing. Plane/cylinder normal pulses occupy the top row; force-ramp sliding and braking occupy the bottom row.",
        ),
    ]
    out = []
    for key, title, ids, caption in groups:
        if any(i not in byid for i in ids):
            continue
        path = MEDIA / f"compare-{key}.mp4"
        if not path.exists():
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
            durations = []
            for i in ids:
                source = DOC / byid[i]["src"]
                command += ["-i", str(source)]
                durations.append(
                    float(
                        subprocess.check_output(
                            [
                                "ffprobe",
                                "-v",
                                "error",
                                "-show_entries",
                                "format=duration",
                                "-of",
                                "csv=p=0",
                                str(source),
                            ],
                            text=True,
                        )
                    )
                )
            chains = [
                f"[{n}:v]crop=640:446:0:0,tpad=stop_mode=clone:stop_duration=30,setpts=PTS-STARTPTS[v{n}]"
                for n in range(4)
            ]
            chains.append("[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|640_0|0_446|640_446[v]")
            command += [
                "-filter_complex",
                ";".join(chains),
                "-map",
                "[v]",
                "-t",
                str(max(durations)),
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-preset",
                "fast",
                "-crf",
                "21",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ]
            subprocess.run(command, check=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(path),
                    "-frames:v",
                    "1",
                    str(path.with_suffix(".jpg")),
                ],
                check=True,
            )
        out.append(
            dict(
                video_id="compare-" + key,
                title=title,
                src=str(path.relative_to(DOC)),
                poster=str(path.with_suffix(".jpg").relative_to(DOC)),
                caption=caption,
                source_ids=ids,
            )
        )
        print(path.name, flush=True)
    write_json(DOC / "data/video_comparisons.json", out)
    return out


def movie(film, caption=""):
    return f'''<figure><video controls muted loop playsinline preload="none" poster="{html.escape(film["poster"])}" style="width:100%;border-radius:10px"><source src="{html.escape(film["src"])}" type="video/mp4"></video><figcaption><strong>{html.escape(film.get("title", film["video_id"]))}</strong>. {html.escape(caption or film.get("caption", ""))} <a href="{html.escape(film["src"])}">MP4</a></figcaption></figure>'''


def summary_section():
    path = DOC / "data/video_comparisons.json"
    if not path.exists():
        return ""
    rows = json.loads(path.read_text())
    body = '<p>Watch synchronized overview comparisons and the full contact close-ups in the <a href="20261004-contact_videos.html">video gallery</a>. The gallery includes retained grasps, slip, release, preload failures and explicitly static snapshots.</p>'
    for key in ("compare-resolution", "compare-payload", "compare-creep"):
        cell = next(r for r in rows if r["video_id"] == key)
        body += movie(cell)
    body += """<p><strong>Video rerun finding:</strong> the previously completed Newton unreduced translated case now fails during preload, both with and without the optional recorder. Its earlier completed summary remains historical evidence, but its successful dynamic transfer is not reproduced. The gallery shows the observed failure; the static Newton pressure tests remain a separate result.</p>"""
    return section("Watch the actual trajectories", body, "videos")


def main():
    films = json.loads((DOC / "data/video_manifest.json").read_text())
    grids = comparisons(films)
    body = section(
        "Start with synchronized comparisons",
        f"""<p>{len(films)} individual clips and {len(grids)} synchronized comparisons cover resolution, mass, preload, payload, friction, timestep, CPU/GPU/Drake, isolated slip, Newton pressure and the 6 mm fillet.</p>
    <p>Watch recorded native motion at 0.2× speed, with contact close-ups in the individual clips below. Failures, frozen end states and static snapshots are labeled in the footage.</p>
    <details><summary>Recording, playback and contact-color details</summary>
    <p>Dynamic footage replays recorded native poses. CPU and Drake were rerun into separate capture directories, and their motion/force summaries agree with the original runs. GPU clips replay the original world-0 qpos records. No motion or forces are interpolated. A shared visual mirror supplies the hand geometry; link translucency and painted tool markers improve visibility without changing simulation.</p>
    <p>Dynamic clips generally play at <strong>0.2× speed</strong>, with simulation time printed in every frame. Very short preload failures use a separately labeled slower speed. Red badges mark a stopped solve or escape guard, and the final recorded pose freezes. A frozen image does not mean the object continued holding. Static pressure/fillet sequences are prescribed snapshots, clearly labeled as having no time integration.</p>
    <p>Sample-sphere colors use recorded native normal force divided by sample area, on a common 0–250 kPa scale. Drake close-ups use native face-centroid pressure glyphs. Newton dynamic and pressure-only glyphs use force on a common 0–0.1 N scale. GPU clips use neutral geometry colors because per-point force records were not saved. The geometry is rigid; these colors do not draw physical mesh deformation.</p>
    </details>
    <p><a href="20261004-articulated_contact_transfer.html">Quantitative report</a> · <a href="data/video_manifest.json">Individual-clip manifest</a> · <a href="data/video_comparisons.json">Comparison manifest</a> · <a href="data/video_capture_validation.json">Native rerun comparisons</a></p>"""
        + "".join(movie(g) for g in grids),
        "comparisons",
    )
    body += section(
        "Newton dynamic replay did not reproduce its earlier completion",
        """<p>The first historical unreduced translated Newton run completed, with unmatched preload. During video capture, the same case now violates the robustness guard during its first 5 ms block. A separate control run without recording also fails. The successful historical trajectory has no saved full-body pose trace, so it cannot be honestly reconstructed as a successful movie. The earlier record is preserved and this discrepancy is explicit.</p><p>The stock reduced native case reproduces its preload failure near 80 ms. Videos below show the states actually observed. Static pressure-generation clips demonstrate that separate component; they do not establish successful Newton dynamics.</p>""",
        "newton-replay",
    )
    categories = [
        ("Resolution, torque and mapping", "transfer"),
        ("Holding, mass and preload", "holding"),
        ("Timestep-refined holding", "holding_refine"),
        ("Friction and creep", "friction_transfer"),
        ("Refined friction", "friction_refine"),
        ("Payload retention and release", "payload"),
        ("Precomputed mapping", "transfer_compiled_fine"),
        ("Drake hydroelastic", "drake_transfer"),
        ("Actual GPU world 0", "mjlab_transfer"),
        ("Newton dynamic limits", "newton_transfer"),
        ("Isolated sliding and load pulses", "slip_fine"),
        ("Newton static pressure and reduction", "newton_pressure"),
        ("6 mm fillet static snapshots", "fillet"),
    ]
    for title, phase in categories:
        clips = [f for f in films if f["phase"] == phase]
        if not clips:
            continue
        content = f"<details><summary>{html.escape(title)} · {len(clips)} clips</summary>"
        for f in clips:
            caption = f"Recorded outcome: {f['status']}. "
            if f["backend"] in ("pressure", "fillet"):
                caption += "Prescribed static snapshots; no integrated dynamics."
            else:
                caption += f"Playback {f['playback_speed']:g}×; last saved state at {f['simulation_end']:.5g} s. "
            if f["backend"] == "newton" and f["run_id"] == "translated_unreduced":
                caption += "Historical completion was not reproduced."
            content += movie(f, caption)
        body += section(title, content + "</details>", phase)
    body += section(
        "Existing full-task demonstration",
        """<p>This original 2 October closed-loop sphere-hand chain video predates the corrected contact-law benchmark. It shows the existing pickup/transfer task with the frozen contact mapping. The <a href="20261004-task_contact_benchmark.html">new SR2 task and geometry benchmark</a> now films the corrected 1/0.5 mm contact variants beside legacy MuJoCo and Drake, then tests <code>diagexact</code> on the same tasks.</p><figure><video controls muted loop playsinline preload="none" style="width:100%"><source src="../20261002-hom_chain/media/20261002-chain_s1_closed.mp4" type="video/mp4"></video><figcaption>Original 1 mm sphere chain, closed loop. <a href="../20261002-hom_chain/media/20261002-chain_s1_closed.mp4">MP4</a> · <a href="../20261002-hom_chain/20261002-hom_screwdriver_chain.html">Original report</a></figcaption></figure>""",
        "task-reference",
    )
    body += section(
        "Reproduce and inspect",
        """<p><a href="README.md">The runbook</a> includes capture and rendering commands. Raw state logs, native contact records, exact configurations, summaries and capture-source hashes live under <code>results/20261004-distributed-contact/video_capture/</code>. MP4s and JPEG posters are portable under <code>media/</code>. Rendering uses forward kinematics of saved state, with no contact re-solve.</p><p>Videos make gross motion, contact location, retention, release and obvious numerical failure inspectable. Their finite state-sampling rate cannot resolve every physics-step oscillation or establish a stiffness coefficient by itself. The raw contact audits and videos remain paired evidence.</p>""",
        "reproduce",
    )
    page = frame(
        "Contact videos: native trajectories and close-ups",
        "Watch the behavior behind the force, holding and speed comparisons.",
        body,
        "Recorded native state · fixed cameras · full failures retained",
    )
    page = page.replace(
        "Generated by <code>scripts/distributed_contact_page.py</code> from the source DOCX and run manifests.",
        "Generated by <code>scripts/distributed_contact_video_page.py</code> from real state captures and MP4s.",
    )
    (DOC / "20261004-contact_videos.html").write_text(retro_style.apply(page))  # plain page style (owner, 2026-10-09)
    print(len(films), "individual clips;", len(grids), "comparisons")


if __name__ == "__main__":
    main()
