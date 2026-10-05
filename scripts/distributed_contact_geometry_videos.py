#!/usr/bin/env python3
"""Film actual short guided sliding trajectories along/across curvature."""

import csv
import json

from PIL import Image

from contact_surface.records import ROOT, write_json
from contact_surface.video import StateCapture, load_states
from distributed_contact_videos import MovieView, capture_isolated, encode, state_at

DOC = ROOT / "docs/experiments/20261004-codex"
RESULTS = ROOT / "results/20261004-distributed-contact/sliding_geometry_guided"
CAPTURE = ROOT / "results/20261004-distributed-contact/video_capture_geometry_guided"
MEDIA = DOC / "media"


def selected():
    for row in json.loads((RESULTS / "manifest.json").read_text()):
        c = row["config"]
        shape, axis, spacing, radius = (
            c["geometry"], c.get("slide_axis", "x"), c["spacing_mm"], c["object_radius_mm"]
        )
        key = (shape, axis, spacing, radius)
        if key not in {
            ("plane", "x", 1.0, 12.5),
            ("cylinder", "x", 1.0, 12.5),
            ("cylinder", "y", 1.0, 12.5),
            ("sphere", "y", 1.0, 25.0),
            ("cylinder", "y", 0.5, 12.5),
        } or c["phase_degrees"] != 0:
            continue
        ref = RESULTS / row["run_id"]
        s = json.loads((ref / "summary.json").read_text())
        if s["status"] != "complete":
            continue
        direction = ("along axis" if axis == "x" and shape == "cylinder"
                     else "across curvature" if axis == "y" and shape == "cylinder"
                     else "tangent " + axis)
        title = f"{shape.title()} R{radius:g} mm | {direction} | {spacing:g} mm samples"
        yield dict(backend="isolated", phase="sliding_geometry_guided", run_id=row["run_id"],
                   video_id=f"geometry-guided-{shape}-{axis}-s{spacing:g}", config=c,
                   title=title, reference=str(ref.relative_to(ROOT)), original=s)
    stress = ROOT / "results/20261004-distributed-contact/sliding_geometry"
    for row in json.loads((stress / "manifest.json").read_text()):
        c = row["config"]
        if (c["geometry"], c["object_radius_mm"], c["slide_axis"], c["spacing_mm"],
                c["phase_degrees"]) not in {
                    ("cylinder", 12.5, "y", 1.0, 0.0),
                    ("sphere", 12.5, "y", 1.0, 0.0),
                }:
            continue
        ref = stress / row["run_id"]
        s = json.loads((ref / "summary.json").read_text())
        if s["status"] != "failed":
            continue
        with (ref / "timeseries.csv").open(newline="") as source:
            last = None
            for record in csv.DictReader(source):
                last = record
        end = float(last["time"]) + c["timestep"]
        yield dict(backend="isolated", phase="sliding_geometry", run_id=row["run_id"],
                   video_id=f"geometry-force-{c['geometry']}-y-r12p5",
                   config=c, title=f"{c['geometry'].title()} R12.5 mm | force ramp | fixture exit",
                   reference=str(ref.relative_to(ROOT)), original=s,
                   stop_time_s=end, stopped=True,
                   reference_end=dict(time_s=end, x_m=float(last["x"]), z_m=float(last["z"])))


def main():
    CAPTURE.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    films = []
    for case in selected():
        path = CAPTURE / case["video_id"]
        path.mkdir(exist_ok=True)
        if not (path / "summary.json").exists():
            state = StateCapture(path, "cpu")
            try:
                summary = capture_isolated(case, path, state)
            finally:
                state.close()
            if case.get("stopped"):
                reference = case["reference_end"]
                summary["reference_comparison"] = {
                    "time_s": dict(reference=reference["time_s"], captured=summary["simulated_s"],
                                   difference=summary["simulated_s"] - reference["time_s"]),
                    "x_m": dict(reference=reference["x_m"], captured=summary["final_tangent_position_m"],
                                difference=summary["final_tangent_position_m"] - reference["x_m"]),
                    "z_m": dict(reference=reference["z_m"], captured=summary["final_height_m"],
                                difference=summary["final_height_m"] - reference["z_m"]),
                }
                summary["reproduced"] = (
                    abs(summary["reference_comparison"]["time_s"]["difference"]) <= case["config"]["timestep"] * 2
                    and abs(summary["reference_comparison"]["x_m"]["difference"]) < 0.00005
                    and abs(summary["reference_comparison"]["z_m"]["difference"]) < 0.00005
                )
            else:
                original = case["original"]
                summary["reference_comparison"] = {
                    k: dict(reference=original.get(k), captured=summary.get(k),
                            difference=summary.get(k, 0) - original.get(k, 0))
                    for k in ("slip_distance_mm", "peak_velocity")
                }
                summary["reproduced"] = all(
                    abs(v["difference"]) < 0.01 for v in summary["reference_comparison"].values()
                )
            write_json(path / "summary.json", summary)
        else:
            summary = json.loads((path / "summary.json").read_text())
        states = load_states(path / "states.jsonl.gz")
        film = MEDIA / (case["video_id"] + ".mp4")
        poster = film.with_suffix(".jpg")
        if not film.exists():
            view = MovieView(dict(case, playback_speed=0.25, stopped=False))

            def frames():
                for k in range(round((states[-1]["time"] / 0.25 + 0.5) * 30)):
                    t = k / 30 * 0.25
                    frame = view.draw(state_at(states, t))
                    if k == 0:
                        Image.fromarray(frame).save(poster)
                    yield frame

            try:
                encode(film, frames())
            finally:
                view.close()
        films.append(dict(video_id=case["video_id"], title=case["title"],
                          src=str(film.relative_to(DOC)), poster=str(poster.relative_to(DOC)),
                          reference=case["reference"], config=case["config"],
                          stopped=bool(case.get("stopped")),
                          reproduced=summary["reproduced"],
                          comparison=summary["reference_comparison"]))
        print(case["video_id"], summary["reproduced"], flush=True)
    write_json(DOC / "data/task_geometry_video_manifest.json", films)


if __name__ == "__main__":
    main()
