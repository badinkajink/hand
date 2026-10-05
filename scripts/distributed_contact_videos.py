#!/usr/bin/env python3
"""Capture real native trajectories and render contact movies without re-solving them."""

import argparse
import json
import math
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from contact_surface.records import ROOT, versions, write_json
from contact_surface.video import StateCapture, load_states

DOC = ROOT / "docs/experiments/20261004-codex"
RESULTS = ROOT / "results/20261004-distributed-contact"
OUT = RESULTS / "video_capture"
MEDIA = DOC / "media"


def cases():
    selected = {
        "cpu": {
            "transfer": [
                "fine_s1.0_tau0.003",
                "fine_s1.0_tau0.012",
                "fine_s0.5_tau0.003",
                "fine_s0.5_tau0.012",
                "refine_s0.25",
                "fine_s1.0_tau0.048",
                "fine_s0.5_tau0.048",
                "fast_refine_s025",
                "coarse_s0.5_dt0.0005",
                "compiled_s1.0",
                "compiled_s0.5",
            ],
            "holding": [
                "hold_s2.0_m1.0",
                "hold_s1.0_m1.0",
                "hold_s0.5_m1.0",
                "hold_s0.5_m4.0",
                "hold_s0.5_m0.25",
                "hold_load4_s0.5",
            ],
            "holding_refine": ["quarter_s0.5_dt1.25e-05"],
            "friction_transfer": ["creep_ct100.0", "creep_ct1000.0"],
            "friction_refine": ["creep_ct1000_dt6.25", "hold_ct100_dt25"],
            "payload": [
                "payload_s1_m1",
                "payload_s05_m1",
                "payload_s1_m8",
                "payload_s05_m8",
                "payload_s1_m16",
                "payload_s05_m16",
                "payload_s1_m32",
                "payload_s05_m32",
            ],
            "transfer_compiled_fine": ["compiled_fine_s1", "compiled_fine_s05"],
        },
        "drake": {
            "drake_transfer": [
                "drake_tau0.003",
                "drake_tau0.012",
                "drake_tau0.048",
                "drake_hold_m1.0",
                "drake_hold_m0.25",
                "drake_hold_m4.0",
                "drake_hold_load4",
            ]
        },
        "newton": {"newton_transfer": ["translated_unreduced", "native_h05"]},
        "gpu": {
            "mjlab_transfer": [
                "fine_s1.0_batch128",
                "fine_s0.5_batch128",
                "s1.0_batch128",
                "s0.5_batch128",
            ]
        },
    }
    rows = []
    for backend, phases in selected.items():
        for phase, ids in phases.items():
            for rid in ids:
                ref = RESULTS / phase / rid
                summary = json.loads((ref / "summary.json").read_text())
                rows.append(
                    dict(
                        backend=backend,
                        phase=phase,
                        run_id=rid,
                        video_id=f"{phase}-{rid}",
                        config=summary["config"],
                        reference=str(ref.relative_to(ROOT)),
                    )
                )
    for item in json.loads((RESULTS / "slip_fine/manifest.json").read_text()):
        c = item["config"]
        if (
            c["mass_scale"] == 1
            and c["spacing_mm"] == 0.5
            and c["timestep"] == 0.00005
            and c["tangent_damping_ratio"] == 10
            and c["group"] == "fine_separated_friction"
        ):
            ref = RESULTS / "slip_fine" / item["run_id"]
            rows.append(
                dict(
                    backend="isolated",
                    phase="slip_fine",
                    run_id=item["run_id"],
                    video_id=f"isolated-{c['geometry']}-{c['scenario']}",
                    config=c,
                    reference=str(ref.relative_to(ROOT)),
                    title=f"Isolated {c['geometry']}: {c['scenario'].replace('_', ' ')}",
                )
            )
    for p in sorted(OUT.glob("snapshots-*.json")):
        rows.extend(json.loads(p.read_text()))
    return rows


def capture_isolated(case, directory, capture):
    from contact_surface.mujoco_model import build
    from contact_surface.runtime import step1
    from contact_surface.slip import apply_wrench
    from contact_surface.metrics import contacts

    c = case["config"]
    slide = 0 if c.get("slide_axis", "x") == "x" else 1
    m, d, p, xml = build(c)
    (directory / "model.xml").write_text(xml)
    d.qpos[2] -= 1e-5
    meta = dict(P=[0, 0, p["radius"]], u=[0, 0, 1], tool=1, object_mass=p["object_mass"])
    peak = 0.0
    start_x = None
    last_normal = 0.0
    for k in range(round(c["duration"] / m.opt.timestep)):
        t = k * m.opt.timestep
        apply_wrench(d, p, c, t)
        if t >= 0.35:
            peak = max(peak, abs(float(d.qvel[slide])))
            if start_x is None:
                start_x = float(d.qpos[slide])
        step1(m, d, p)
        mujoco.mj_step2(m, d)
        if capture.due(float(d.time)):
            rows = contacts(m, d, p, t)
            last_normal = sum(r["f_n"] for r in rows)
            record = dict(
                time=float(d.time),
                normal_total=last_normal,
                x=float(d.qpos[slide]),
                vx=float(d.qvel[slide]),
                stage=c["scenario"],
            )
            capture.cpu(m, d, meta, rows, record)
        if case.get("stop_time_s") is not None and float(d.time) >= case["stop_time_s"]:
            break
    return dict(
        status="stopped_at_guard" if case.get("stop_time_s") is not None else "complete",
        config=c,
        peak_velocity=peak,
        slip_distance_mm=(float(d.qpos[slide]) - start_x) * 1000,
        final_normal_force=last_normal,
        final_tangent_position_m=float(d.qpos[slide]),
        final_height_m=float(d.qpos[2]),
        simulated_s=float(d.time),
    )


def capture(backend, limit=None):
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "manifest.json", cases())
    write_json(OUT / f"versions-{backend}.json", versions())
    launched = 0
    for case in cases():
        if case["backend"] != backend:
            continue
        directory = OUT / case["video_id"]
        directory.mkdir(exist_ok=True)
        if (directory / "summary.json").exists():
            continue
        if limit is not None and launched >= limit:
            break
        write_json(directory / "case.json", case)
        state = StateCapture(directory, "cpu" if backend == "isolated" else backend)
        try:
            if backend == "cpu":
                from distributed_contact_transfer import run_cpu

                summary = run_cpu(case["config"], directory, on_frame=state)
            elif backend == "drake":
                from distributed_contact_drake import run

                summary = run(case["config"], directory, on_frame=state)
            elif backend == "newton":
                from distributed_contact_newton import run

                summary = run(case["config"], directory, on_frame=state)
            elif backend == "isolated":
                summary = capture_isolated(case, directory, state)
            else:
                raise ValueError(backend)
            reference = json.loads((ROOT / case["reference"] / "summary.json").read_text())
            comparison = {}
            for key in (
                "peak_angular_speed",
                "peak_displacement_mm",
                "normal_L1_relative",
                "simulated_s",
                "peak_velocity",
                "slip_distance_mm",
                "final_normal_force",
            ):
                if summary.get(key) is not None and reference.get(key) is not None:
                    comparison[key] = dict(
                        reference=reference[key],
                        captured=summary[key],
                        difference=summary[key] - reference[key],
                    )
            summary["reference_comparison"] = comparison
            summary["reference_status"] = reference["status"]
            summary["purpose"] = (
                "visual replay; capture overhead excluded from old benchmark; original records preserved"
            )
            write_json(directory / "summary.json", summary)
            print(case["video_id"], summary["status"], state.count, flush=True)
        finally:
            state.close()
        launched += 1


def heat(value, maximum=250000.0):
    u = np.clip(value / maximum, 0, 1)
    cold, warm, hot = (
        np.array([0.4, 0.58, 0.72]),
        np.array([1.0, 0.72, 0.15]),
        np.array([0.8, 0.1, 0.08]),
    )
    c = cold * (1 - 2 * u) + warm * (2 * u) if u < 0.5 else warm * (2 - 2 * u) + hot * (2 * u - 1)
    return [*c, 1.0]


class MovieView:
    def __init__(self, case):
        from distributed_contact_transfer import build

        self.case = case
        self.isolated = case["backend"] in ("isolated", "fillet")
        if case["backend"] == "fillet":
            self.m = mujoco.MjModel.from_xml_path(str(ROOT / case["reference"] / "model.xml"))
            self.d = mujoco.MjData(self.m)
            self.meta = dict(P=[0, 0, 0.003], u=[0, 0, 1], tool=1)
        elif self.isolated:
            from contact_surface.mujoco_model import build as iso_build

            self.m, self.d, p, _ = iso_build(case["config"])
            self.meta = dict(P=[0, 0, p["radius"]], u=[0, 0, 1], tool=1)
        else:
            self.m, self.d, _, self.meta = build(dict(case["config"], policy="compiled"))
        self.r = mujoco.Renderer(self.m, 360, 640, max_geom=20000)
        self.rc = mujoco.Renderer(self.m, 360, 360, max_geom=20000)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = np.array(self.meta["P"]) + [0, 0, 0.015]
        self.cam.distance = 0.25
        self.cam.azimuth = math.degrees(math.atan2(self.meta["u"][1], self.meta["u"][0]))
        self.cam.elevation = -12
        self.close_cam = mujoco.MjvCamera()
        self.close_cam.distance = 0.035
        self.close_cam.azimuth = self.cam.azimuth + 30
        self.close_cam.elevation = -20
        if self.isolated:
            self.cam.lookat[:] = [0, 0, 0.022]
            self.cam.distance = 0.24 if case["config"]["geometry"] == "plane" else 0.15
            self.cam.azimuth, self.cam.elevation = 100, -20
            self.close_cam.lookat[:] = [0, 0, 0.011]
            self.close_cam.azimuth, self.close_cam.elevation = 90, -10
            self.close_cam.distance = 0.05
        if case["backend"] == "fillet":
            self.cam.lookat[:] = [0, 0, 0.01]
            self.cam.distance = 0.08
            self.cam.azimuth, self.cam.elevation = 135, -35
            self.close_cam.lookat[:] = [0, 0, 0.009]
            self.close_cam.azimuth, self.close_cam.elevation = 135, -35
            self.close_cam.distance = 0.045
            self.m.vis.headlight.ambient[:] = 0.5
            self.m.vis.headlight.diffuse[:] = 0.6
        self.rgba = self.m.geom_rgba.copy()
        self.pads = [
            g
            for g in range(self.m.ngeom)
            if self.m.geom(g).name.startswith(("thumb_pad", "index_pad", "sample_"))
        ]
        self.tips = [
            g
            for g in range(self.m.ngeom)
            if self.m.geom(g).name in ("thumb_tipsphere", "index_tipsphere")
        ]
        self.tool_geoms = [
            g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == self.meta["tool"]
        ]
        self.hide_close = [
            g
            for g in range(self.m.ngeom)
            if self.m.body(self.m.geom_bodyid[g]).name.startswith(("index_", "middle_"))
        ]
        self.robot_geoms = [
            g
            for g in range(self.m.ngeom)
            if g not in self.pads and g not in self.tool_geoms and self.m.geom_bodyid[g] > 0
        ]
        self.hydro = case["backend"] in ("drake", "newton", "pressure")

    def glyph(self, scene, pos, normal, size, color):
        from hom_contact_rig import _rot_z_to

        if scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.array([size, size, 0.000035]),
            np.array(pos),
            _rot_z_to(normal).ravel(),
            np.array(color, dtype=np.float32),
        )
        scene.ngeom += 1

    def draw(self, state, overview=False):
        self.d.qpos[:] = state["qpos"]
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)
        self.m.geom_rgba[:] = self.rgba
        if self.case["backend"] == "fillet":
            self.m.geom_rgba[self.tool_geoms, 3] = 0.12
        if not self.isolated:
            self.m.geom_rgba[self.robot_geoms, 3] = 0.4
            self.m.geom_rgba[self.m.geom("tool").id] = [0.16, 0.38, 0.60, 1]
        points = state.get("contacts", [])
        if self.hydro:
            self.m.geom_rgba[self.pads, 3] = 0
            self.m.geom_rgba[self.tips, 3] = 0.5
        else:
            self.m.geom_rgba[self.tips, 3] = 0.10
            self.m.geom_rgba[self.pads] = [0.65, 0.7, 0.74, 1]
            for point in points:
                self.m.geom_rgba[int(point[9])] = heat(point[7])
        self.r.update_scene(self.d, self.cam)
        wide = self.r.render().copy()
        if not self.isolated:
            self.close_cam.lookat[:] = self.d.xpos[self.m.body("thumb_tip").id]
            n = self.d.xpos[self.meta["tool"]] - self.close_cam.lookat
            self.close_cam.azimuth = math.degrees(math.atan2(n[1], n[0])) + 20
            self.close_cam.elevation = -10
            self.m.geom_rgba[self.robot_geoms, 3] = 0.03
            self.m.geom_rgba[self.tips, 3] = 0.05
        self.m.geom_rgba[self.tool_geoms, 3] = 0.06 if self.case["backend"] == "fillet" else 0.20
        self.m.geom_rgba[self.hide_close, 3] = 0.03
        self.rc.update_scene(self.d, self.close_cam)
        if self.hydro:
            center = self.close_cam.lookat
            for p in points:
                if np.linalg.norm(np.array(p[:3]) - center) > 0.018:
                    continue
                if self.case["backend"] == "drake":
                    color, radius = heat(p[7]), np.sqrt(max(p[8], 1e-12) / np.pi)
                elif self.case["backend"] == "pressure":
                    color, radius = heat(p[6], 0.1), 0.0004
                else:
                    color, radius = heat(p[6], 0.1), 0.0006
                self.glyph(self.rc.scene, p[:3], p[3:6], radius, color)
        close = self.rc.render().copy()
        self.m.geom_rgba[:] = self.rgba
        from PIL import Image, ImageDraw, ImageFont

        h, w = (446, 640) if overview else (496, 1000)
        im = Image.new("RGB", (w, h), (246, 247, 248))
        im.paste(Image.fromarray(wide), (0, 65))
        if not overview:
            im.paste(Image.fromarray(close), (640, 65))
        d = ImageDraw.Draw(im)
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
        small = ImageFont.truetype("DejaVuSansMono.ttf", 14)
        cfg, metrics = self.case["config"], state.get("metrics", {})
        title = self.case.get("title", self.case["run_id"])
        d.text((10, 7), title, fill=(20, 26, 32), font=font)
        line = f"{self.case['backend']} | dt {cfg.get('timestep', 0) * 1e6:g} us | mass {24.5437 * cfg.get('mass_scale', 1):.1f} g"
        if "spacing_mm" in cfg:
            line += f" | spacing {cfg['spacing_mm']:g} mm"
        if self.case["backend"] in ("pressure", "fillet"):
            line = "Static snapshots of native contact data; no time integration"
        d.text((10, 34), line, fill=(45, 55, 65), font=small)
        normal = metrics.get(
            "normal_total", metrics.get("normal_thumb", 0) + metrics.get("normal_index", 0)
        )
        omega = metrics.get("angular_velocity", 0)
        displ = metrics.get("displacement_mm", metrics.get("displacement_m", 0) * 1000)
        stop = metrics.get("stage") == "stopped"
        normal_text = (
            f"{normal:.3f}"
            if any(k in metrics for k in ("normal_total", "normal_thumb"))
            else "unlogged"
        )
        omega_text = f"{omega:+.3f}" if "angular_velocity" in metrics else "unlogged"
        displ_text = (
            f"{displ:.2f}"
            if any(k in metrics for k in ("displacement_mm", "displacement_m"))
            else "unlogged"
        )
        line = f"t={state['time']:.4f} s  N={normal_text}  omega={omega_text} rad/s  move={displ_text} mm"
        if not self.isolated and self.case["backend"] != "pressure":
            from distributed_contact_transfer import command

            tau = metrics.get("torque", command(state["time"], cfg)) * 1000
            line += f"  tau_cmd={tau:+.2f} mNm"
        if self.isolated:
            axis = self.case["config"].get("slide_axis", "x")
            line = f"t={state['time']:.4f} s  N={normal:.3f}  {axis}={metrics.get('x', 0) * 1000:+.2f} mm  v{axis}={metrics.get('vx', 0):+.3f} m/s"
        if self.case["backend"] == "pressure":
            line = f"Prescribed indentation {metrics['indentation_mm']:g} mm | normal sum {normal:.3f} N | {metrics['contacts']} points"
        if self.case["backend"] == "fillet":
            line = f"Prescribed tilt {metrics['tilt_degrees']:g} deg | normal sum {normal:.3f} N | 0.2 mm indentation"
        d.text((10, 426), line, fill=(25, 30, 35), font=small)
        if not overview:
            info = (
                "thumb close-up; tool/index translucent | native pressure: blue -> red, 0-250 kPa"
            )
            if self.case["backend"] == "newton":
                info = "thumb close-up; solver point force: blue -> red, 0-0.1 N"
            if self.isolated:
                info = "sample-pad close-up; object translucent | native pressure: blue -> red, 0-250 kPa"
            if self.case["backend"] == "gpu":
                info = (
                    "actual GPU world 0; neutral geometry colors; per-point pressure not recorded"
                )
            if self.case["backend"] == "pressure":
                info = "Newton pressure quadrature output; point force: blue -> red, 0-0.1 N"
            if self.case["backend"] == "fillet":
                info = "Analytic 6 mm front fillet; native static reaction; incomplete hardware CAD"
            d.text((10, 448), info, fill=(55, 60, 65), font=small)
            d.text(
                (10, 473),
                "Prescribed static snapshots; no interpolated forces"
                if self.case["backend"] in ("pressure", "fillet")
                else f"playback {self.case.get('playback_speed', 0.2):g}x | recorded state samples; no interpolation",
                fill=(55, 60, 65),
                font=small,
            )
        if stop or state.get("frozen"):
            text = (
                "STOPPED AT GUARD" if stop or self.case.get("stopped") else "RECORDING END (FROZEN)"
            )
            d.rectangle(
                [w - 260, 3, w, 29],
                fill=(122, 30, 30) if self.case.get("stopped") else (45, 60, 75),
            )
            d.text((w - 251, 7), text, fill="white", font=small)
        return np.asarray(im)

    def close(self):
        self.r.close()
        self.rc.close()


def state_at(states, t):
    times = np.array([s["time"] for s in states])
    ix = max(0, int(np.searchsorted(times, t, side="right")) - 1)
    s = dict(states[ix])
    if t > times[-1] + 1e-9:
        s["frozen"] = True
    return s


def encode(path, frames, fps=30):
    first = next(frames)
    h, w = first.shape[:2]
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
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
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        proc.stdin.write(first.tobytes())
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError(f"ffmpeg failed for {path}")


def render(limit=None, force=False):
    MEDIA.mkdir(exist_ok=True)
    films = []
    for case in cases():
        directory = OUT / case["video_id"]
        if case["backend"] == "gpu":
            records = json.loads((ROOT / case["reference"] / "trace.json").read_text())
            states = [
                dict(
                    time=0.0,
                    qpos=json.loads((ROOT / case["reference"] / "summary.json").read_text())[
                        "fixture"
                    ]["initial_qpos"],
                    contacts=[],
                    metrics={},
                )
            ]
            states += [
                dict(time=r["time"], qpos=r["qpos"], contacts=[], metrics=r) for r in records
            ]
            case = dict(case, title=case["run_id"] + " (actual GPU world 0; geometry view)")
            # No stored per-point forces: retain neutral sample colors.
            summary = json.loads((ROOT / case["reference"] / "summary.json").read_text())
        elif (directory / "summary.json").exists():
            summary = json.loads((directory / "summary.json").read_text())
            states = load_states(directory / "states.jsonl.gz")
        else:
            continue
        if not states:
            continue
        case = dict(case, stopped=summary["status"] != "complete")
        path = MEDIA / f"{case['video_id']}.mp4"
        duration = case["config"].get("duration", 0.8)
        speed = 0.2
        if case["backend"] in ("pressure", "fillet"):
            duration = states[-1]["time"] + 0.3
        if states[-1]["time"] < 0.1:
            duration = states[-1]["time"]
            speed = max(duration / 4.0, 0.0001)
        case = dict(case, playback_speed=speed)
        if force or not path.exists():
            view = MovieView(case)

            def frames():
                for k in range(round((duration / speed + 1.0) * 30)):
                    t = min(k / 30 * speed, duration + 0.001)
                    frame = view.draw(state_at(states, t))
                    if k == 0:
                        from PIL import Image

                        Image.fromarray(frame).save(path.with_suffix(".jpg"))
                    yield frame

            try:
                encode(path, frames())
            finally:
                view.close()
        films.append(
            dict(
                **case,
                src=str(path.relative_to(DOC)),
                poster=str(path.with_suffix(".jpg").relative_to(DOC)),
                status=summary["status"],
                simulation_end=states[-1]["time"]
                if case["backend"] not in ("pressure", "fillet")
                else None,
                source="original GPU states"
                if case["backend"] == "gpu"
                else "native prescribed static snapshots"
                if case["backend"] in ("pressure", "fillet")
                else "native rerun captured separately",
                reference_comparison=summary.get("reference_comparison"),
            )
        )
        print(path.name, flush=True)
        if limit and len(films) >= limit:
            break
    write_json(DOC / "data/video_manifest.json", films)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["capture", "render"])
    p.add_argument("--backend", choices=["cpu", "drake", "newton", "isolated"])
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    if a.action == "capture":
        capture(a.backend, a.limit)
    else:
        render(a.limit, a.force)


if __name__ == "__main__":
    main()
