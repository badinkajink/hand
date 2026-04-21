from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from morphohand.optimization.phase1_grasp import (  # noqa: E402
    Phase1EvalConfig,
    Phase1GraspEvaluator,
    Phase1OptimizationConfig,
    optimize_finger_controls,
)
from morphohand.tools.morphology_xml import (  # noqa: E402
    MorphologyValues,
    apply_morphology_to_qpos,
    extract_morphology_from_qpos,
)


@dataclass(frozen=True)
class FoundationalPose:
    label: str
    finger_ctrl: np.ndarray
    score: float


@dataclass(frozen=True)
class MorphologyBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    len_min: float
    len_max: float


@dataclass(frozen=True)
class TaskResult:
    keyframe: str
    chosen_label: str
    chosen_ctrl: np.ndarray
    score: float
    metrics: dict[str, float]
    feasible: bool
    sparse_adapt_seconds: float
    interval_adapt_seconds: float
    interval_triggered: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run6 combined multitask sampler: evaluate 3 screwdriver keyframes per morphology, "
            "combine sparse+interval adaptation in one run, track rolling efficiency, and render top-5 gifs."
        )
    )
    parser.add_argument(
        "--scene-xml",
        type=Path,
        default=PROJECT_ROOT / "assets" / "mjcf" / "scene_screwdriver_medium.xml",
    )
    parser.add_argument(
        "--keyframes",
        nargs="+",
        default=["open_flat", "open_vertical", "open_90vertical"],
    )
    parser.add_argument(
        "--foundational-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase1" / "run6_foundational",
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--fp-refresh-interval", type=int, default=50)
    parser.add_argument("--morph-sort", choices=["none", "distance"], default="distance")
    parser.add_argument("--x-perturb", type=float, default=0.012)
    parser.add_argument("--y-perturb", type=float, default=0.012)
    parser.add_argument("--len-perturb", type=float, default=0.012)
    parser.add_argument("--x-min", type=float, default=-0.03)
    parser.add_argument("--x-max", type=float, default=0.03)
    parser.add_argument("--y-min", type=float, default=-0.03)
    parser.add_argument("--y-max", type=float, default=0.03)
    parser.add_argument("--len-min", type=float, default=0.0)
    parser.add_argument("--len-max", type=float, default=0.035)
    parser.add_argument("--max-mean-tip-distance", type=float, default=0.022)
    parser.add_argument("--min-contacts", type=float, default=2.0)
    parser.add_argument("--settle-steps", type=int, default=240)
    parser.add_argument("--lift-steps", type=int, default=220)
    parser.add_argument("--hold-steps", type=int, default=140)
    parser.add_argument("--lift-delta-z", type=float, default=0.05)
    parser.add_argument("--lift-ramp-steps", type=int, default=100)
    parser.add_argument("--objective-weight-min-finger-persistence", type=float, default=2.4)
    parser.add_argument(
        "--objective-weight-finger-persistence-imbalance-penalty",
        type=float,
        default=1.2,
    )
    parser.add_argument("--objective-weight-finger-yaw-drift-penalty", type=float, default=1.0)
    parser.add_argument("--objective-weight-finger-flex-drift-penalty", type=float, default=0.5)
    parser.add_argument("--interval-adapt-iterations", type=int, default=12)
    parser.add_argument("--interval-adapt-population", type=int, default=24)
    parser.add_argument("--interval-adapt-elite-fraction", type=float, default=0.25)
    parser.add_argument("--interval-adapt-sigma-init", type=float, default=0.08)
    parser.add_argument("--sparse-adapt-iterations", type=int, default=1)
    parser.add_argument("--sparse-adapt-population", type=int, default=5)
    parser.add_argument("--sparse-adapt-elite-fraction", type=float, default=0.25)
    parser.add_argument("--sparse-adapt-sigma-init", type=float, default=0.06)
    parser.add_argument("--adapt-seed", type=int, default=0)
    parser.add_argument("--top-k-gifs", type=int, default=5)
    parser.add_argument("--gif-width", type=int, default=720)
    parser.add_argument("--gif-height", type=int, default=540)
    parser.add_argument("--gif-fps", type=int, default=25)
    parser.add_argument("--gif-frame-stride", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "phase1")
    parser.add_argument("--tag", default="run6_combined_1000")
    return parser


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_keyframe_qpos(scene_xml: Path, keyframe_name: str) -> list[float]:
    root = ET.parse(scene_xml).getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        raise ValueError(f"No <keyframe> section in {scene_xml}")
    for key in keyframe.findall("key"):
        if key.get("name") == keyframe_name:
            qpos_raw = key.get("qpos") or ""
            values = [float(v) for v in qpos_raw.replace("\n", " ").split()]
            if len(values) < 31:
                raise ValueError(f"Keyframe '{keyframe_name}' in {scene_xml} has short qpos ({len(values)})")
            return values
    raise ValueError(f"Keyframe '{keyframe_name}' not found in {scene_xml}")


def _extract_keyframe_morphology(scene_xml: Path, keyframe_name: str) -> MorphologyValues:
    qpos = _read_keyframe_qpos(scene_xml=scene_xml, keyframe_name=keyframe_name)
    return extract_morphology_from_qpos(qpos=qpos, has_scene_prefix=True)


def _write_scene_with_morphology(base_scene_xml: Path, output_scene_xml: Path, morphology: MorphologyValues) -> None:
    root = ET.parse(base_scene_xml).getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        raise ValueError(f"No <keyframe> section in {base_scene_xml}")

    for key in keyframe.findall("key"):
        qpos_raw = key.get("qpos")
        if not qpos_raw:
            continue
        qpos = [float(v) for v in qpos_raw.replace("\n", " ").split()]
        if len(qpos) < 31:
            continue
        apply_morphology_to_qpos(qpos=qpos, morphology=morphology, has_scene_prefix=True)
        key.set("qpos", "\n        " + " ".join(f"{v:.10g}" for v in qpos) + "\n      ")

    root.set("model", output_scene_xml.stem)
    ET.indent(root, space="  ")
    output_scene_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_scene_xml, encoding="utf-8", xml_declaration=False)


def _morph_to_array(m: MorphologyValues) -> np.ndarray:
    return np.array(
        [
            m.thumb_x,
            m.thumb_y,
            m.thumb_len,
            m.index_x,
            m.index_y,
            m.index_len,
            m.middle_x,
            m.middle_y,
            m.middle_len,
        ],
        dtype=np.float64,
    )


def _morph_distance(a: MorphologyValues, b: MorphologyValues) -> float:
    return float(np.linalg.norm(_morph_to_array(a) - _morph_to_array(b)))


def _clip_morphology(values: MorphologyValues, bounds: MorphologyBounds) -> MorphologyValues:
    return MorphologyValues(
        thumb_x=float(np.clip(values.thumb_x, bounds.x_min, bounds.x_max)),
        thumb_y=float(np.clip(values.thumb_y, bounds.y_min, bounds.y_max)),
        thumb_len=float(np.clip(values.thumb_len, bounds.len_min, bounds.len_max)),
        index_x=float(np.clip(values.index_x, bounds.x_min, bounds.x_max)),
        index_y=float(np.clip(values.index_y, bounds.y_min, bounds.y_max)),
        index_len=float(np.clip(values.index_len, bounds.len_min, bounds.len_max)),
        middle_x=float(np.clip(values.middle_x, bounds.x_min, bounds.x_max)),
        middle_y=float(np.clip(values.middle_y, bounds.y_min, bounds.y_max)),
        middle_len=float(np.clip(values.middle_len, bounds.len_min, bounds.len_max)),
    )


def _sample_morphologies(
    base: MorphologyValues,
    sample_count: int,
    rng: np.random.Generator,
    bounds: MorphologyBounds,
    x_perturb: float,
    y_perturb: float,
    len_perturb: float,
) -> list[MorphologyValues]:
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")

    def key_for(v: MorphologyValues) -> tuple[float, ...]:
        return (
            round(v.thumb_x, 6),
            round(v.thumb_y, 6),
            round(v.thumb_len, 6),
            round(v.index_x, 6),
            round(v.index_y, 6),
            round(v.index_len, 6),
            round(v.middle_x, 6),
            round(v.middle_y, 6),
            round(v.middle_len, 6),
        )

    candidates: list[MorphologyValues] = [base]
    seen: set[tuple[float, ...]] = {key_for(base)}

    while len(candidates) < sample_count:
        proposal = MorphologyValues(
            thumb_x=base.thumb_x + float(rng.uniform(-x_perturb, x_perturb)),
            thumb_y=base.thumb_y + float(rng.uniform(-y_perturb, y_perturb)),
            thumb_len=base.thumb_len + float(rng.uniform(-len_perturb, len_perturb)),
            index_x=base.index_x + float(rng.uniform(-x_perturb, x_perturb)),
            index_y=base.index_y + float(rng.uniform(-y_perturb, y_perturb)),
            index_len=base.index_len + float(rng.uniform(-len_perturb, len_perturb)),
            middle_x=base.middle_x + float(rng.uniform(-x_perturb, x_perturb)),
            middle_y=base.middle_y + float(rng.uniform(-y_perturb, y_perturb)),
            middle_len=base.middle_len + float(rng.uniform(-len_perturb, len_perturb)),
        )
        proposal = _clip_morphology(proposal, bounds)
        key = key_for(proposal)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(proposal)

    return candidates


def _load_foundational_poses(foundational_run_dir: Path, keyframe_name: str) -> list[FoundationalPose]:
    poses: list[FoundationalPose] = []
    for subdir in sorted(foundational_run_dir.iterdir()):
        if not subdir.is_dir():
            continue
        summary_path = subdir / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(payload.get("keyframe")) != keyframe_name:
            continue
        poses.append(
            FoundationalPose(
                label=subdir.name,
                finger_ctrl=np.asarray(payload["best_finger_ctrl"], dtype=np.float64),
                score=float(payload.get("best_score", float("-inf"))),
            )
        )

    if not poses:
        raise ValueError(f"No foundational summaries for keyframe '{keyframe_name}' under {foundational_run_dir}")

    return sorted(poses, key=lambda p: p.score, reverse=True)


def _is_feasible(metrics: dict[str, float], max_mean_tip_distance: float, min_contacts: float) -> bool:
    return (
        float(metrics.get("mean_tip_distance", np.inf)) <= max_mean_tip_distance
        and float(metrics.get("cube_tip_contacts", 0.0)) >= min_contacts
    )


def _adapt_ctrl(
    evaluator: Phase1GraspEvaluator,
    initial_ctrl: np.ndarray,
    cfg: Phase1OptimizationConfig,
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    refined = optimize_finger_controls(evaluator=evaluator, cfg=cfg, initial_finger_ctrl=initial_ctrl)
    elapsed = float(time.perf_counter() - t0)
    return np.asarray(refined["best_finger_ctrl"], dtype=np.float64), elapsed


def _select_best(
    evals: list[tuple[str, np.ndarray, float, dict[str, float], bool]],
) -> tuple[str, np.ndarray, float, dict[str, float], bool]:
    feasible_hits = [e for e in evals if e[4]]
    pool = feasible_hits if feasible_hits else evals
    return max(pool, key=lambda e: e[2])


def _rolling_efficiency(
    rows: list[dict[str, object]],
    window: int,
) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    n = len(rows)
    if n == 0:
        return out

    total_tasks = float(len(json.loads(str(rows[0]["task_score_json"]))))

    for start in range(0, n, window):
        chunk = rows[start : start + window]
        sample_count = len(chunk)
        feasible_task_sum = float(sum(int(r["feasible_task_count"]) for r in chunk))
        all_feasible_sum = float(sum(1 for r in chunk if bool(r["all_tasks_feasible"])))
        out.append(
            {
                "window_start": float(start),
                "window_end": float(start + sample_count - 1),
                "samples_in_window": float(sample_count),
                "rolling_task_feasibility_efficiency": feasible_task_sum / max(1.0, sample_count * total_tasks),
                "rolling_all_task_feasibility_rate": all_feasible_sum / max(1.0, sample_count),
            }
        )
    return out


def _plot_rolling_efficiency(rolling_rows: list[dict[str, float]], out_dir: Path) -> None:
    if not rolling_rows:
        return
    centers = np.asarray(
        [(r["window_start"] + r["window_end"]) * 0.5 for r in rolling_rows],
        dtype=np.float64,
    )
    task_eff = np.asarray([r["rolling_task_feasibility_efficiency"] for r in rolling_rows], dtype=np.float64)
    all_eff = np.asarray([r["rolling_all_task_feasibility_rate"] for r in rolling_rows], dtype=np.float64)

    plt.figure(figsize=(8.0, 4.8))
    plt.plot(centers, task_eff, marker="o", label="Task feasibility efficiency")
    plt.plot(centers, all_eff, marker="s", label="All-task feasibility rate")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Sample index (window center)")
    plt.ylabel("Efficiency")
    plt.title("Run6 rolling sampling efficiency")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rolling_efficiency.png", dpi=180)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    run_t0 = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    tag = args.tag or datetime.now().strftime("run6_combined_%Y%m%d_%H%M%S")
    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = MorphologyBounds(
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        len_min=args.len_min,
        len_max=args.len_max,
    )

    eval_cfg = Phase1EvalConfig(
        settle_steps=args.settle_steps,
        lift_steps=args.lift_steps,
        hold_steps=args.hold_steps,
        lift_delta_z=args.lift_delta_z,
        lift_ramp_steps=args.lift_ramp_steps,
        objective_weight_min_finger_persistence=args.objective_weight_min_finger_persistence,
        objective_weight_finger_persistence_imbalance_penalty=args.objective_weight_finger_persistence_imbalance_penalty,
        objective_weight_finger_yaw_drift_penalty=args.objective_weight_finger_yaw_drift_penalty,
        objective_weight_finger_flex_drift_penalty=args.objective_weight_finger_flex_drift_penalty,
    )
    interval_cfg = Phase1OptimizationConfig(
        iterations=args.interval_adapt_iterations,
        population=args.interval_adapt_population,
        elite_fraction=args.interval_adapt_elite_fraction,
        sigma_init=args.interval_adapt_sigma_init,
        seed=args.adapt_seed,
    )
    sparse_cfg = Phase1OptimizationConfig(
        iterations=args.sparse_adapt_iterations,
        population=args.sparse_adapt_population,
        elite_fraction=args.sparse_adapt_elite_fraction,
        sigma_init=args.sparse_adapt_sigma_init,
        seed=args.adapt_seed,
    )

    foundational_by_keyframe: dict[str, list[FoundationalPose]] = {}
    interval_ctrl_by_keyframe: dict[str, np.ndarray] = {}
    base_morph = _extract_keyframe_morphology(args.scene_xml, keyframe_name=args.keyframes[0])

    for keyframe in args.keyframes:
        poses = _load_foundational_poses(args.foundational_root / keyframe, keyframe)
        foundational_by_keyframe[keyframe] = poses
        interval_ctrl_by_keyframe[keyframe] = np.asarray(poses[0].finger_ctrl, dtype=np.float64).copy()

    candidates = _sample_morphologies(
        base=base_morph,
        sample_count=args.samples,
        rng=rng,
        bounds=bounds,
        x_perturb=args.x_perturb,
        y_perturb=args.y_perturb,
        len_perturb=args.len_perturb,
    )
    if args.morph_sort == "distance":
        candidates.sort(key=lambda m: _morph_distance(m, base_morph))

    gen_dir = out_dir / "generated_mjcf"
    gen_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []

    interval_adapt_count = {k: 0 for k in args.keyframes}
    interval_adapt_seconds = {k: 0.0 for k in args.keyframes}
    sparse_adapt_seconds = {k: 0.0 for k in args.keyframes}
    task_eval_seconds = {k: 0.0 for k in args.keyframes}

    for idx, morphology in enumerate(candidates):
        suffix = (
            f"t{morphology.thumb_x:+0.4f}_{morphology.thumb_y:+0.4f}_{morphology.thumb_len:+0.4f}_"
            f"i{morphology.index_x:+0.4f}_{morphology.index_y:+0.4f}_{morphology.index_len:+0.4f}_"
            f"m{morphology.middle_x:+0.4f}_{morphology.middle_y:+0.4f}_{morphology.middle_len:+0.4f}"
        ).replace("+", "p").replace("-", "n").replace(".", "d")
        scene_xml = gen_dir / f"scene_multi_{suffix}.xml"
        _write_scene_with_morphology(args.scene_xml, scene_xml, morphology)

        task_results: list[TaskResult] = []

        for keyframe in args.keyframes:
            evaluator = Phase1GraspEvaluator(scene_xml=scene_xml, keyframe=keyframe, cfg=eval_cfg)
            interval_ctrl = interval_ctrl_by_keyframe[keyframe]

            interval_triggered = (idx % max(1, args.fp_refresh_interval)) == 0
            interval_secs = 0.0
            if interval_triggered:
                interval_ctrl, interval_secs = _adapt_ctrl(evaluator, interval_ctrl, interval_cfg)
                interval_ctrl_by_keyframe[keyframe] = interval_ctrl.copy()
                interval_adapt_count[keyframe] += 1
                interval_adapt_seconds[keyframe] += interval_secs

            sparse_ctrl, sparse_secs = _adapt_ctrl(evaluator, interval_ctrl, sparse_cfg)
            sparse_adapt_seconds[keyframe] += sparse_secs

            t_eval = time.perf_counter()
            evals: list[tuple[str, np.ndarray, float, dict[str, float], bool]] = []

            s_interval, m_interval = evaluator.evaluate(interval_ctrl)
            f_interval = _is_feasible(m_interval, args.max_mean_tip_distance, args.min_contacts)
            evals.append(("interval", interval_ctrl, s_interval, m_interval, f_interval))

            s_sparse, m_sparse = evaluator.evaluate(sparse_ctrl)
            f_sparse = _is_feasible(m_sparse, args.max_mean_tip_distance, args.min_contacts)
            evals.append(("sparse", sparse_ctrl, s_sparse, m_sparse, f_sparse))

            for pose in foundational_by_keyframe[keyframe]:
                s_pose, m_pose = evaluator.evaluate(pose.finger_ctrl)
                f_pose = _is_feasible(m_pose, args.max_mean_tip_distance, args.min_contacts)
                evals.append((pose.label, np.asarray(pose.finger_ctrl, dtype=np.float64), s_pose, m_pose, f_pose))

            task_eval_seconds[keyframe] += float(time.perf_counter() - t_eval)
            best = _select_best(evals)
            task_result = TaskResult(
                keyframe=keyframe,
                chosen_label=best[0],
                chosen_ctrl=best[1],
                score=float(best[2]),
                metrics=best[3],
                feasible=bool(best[4]),
                sparse_adapt_seconds=float(sparse_secs),
                interval_adapt_seconds=float(interval_secs),
                interval_triggered=bool(interval_triggered),
            )
            task_results.append(task_result)

            task_rows.append(
                {
                    "candidate_id": idx,
                    "scene_xml": str(scene_xml),
                    "keyframe": keyframe,
                    "chosen_label": task_result.chosen_label,
                    "score": task_result.score,
                    "feasible": task_result.feasible,
                    "cube_lift": float(task_result.metrics.get("cube_lift", 0.0)),
                    "cube_tip_contacts": float(task_result.metrics.get("cube_tip_contacts", 0.0)),
                    "mean_tip_distance": float(task_result.metrics.get("mean_tip_distance", 0.0)),
                    "cube_xy_drift": float(task_result.metrics.get("cube_xy_drift", 0.0)),
                    "finger_flex_drift": float(task_result.metrics.get("finger_flex_drift", 0.0)),
                    "interval_triggered": task_result.interval_triggered,
                    "interval_adapt_seconds": task_result.interval_adapt_seconds,
                    "sparse_adapt_seconds": task_result.sparse_adapt_seconds,
                }
            )

        feasible_task_count = int(sum(1 for t in task_results if t.feasible))
        all_tasks_feasible = feasible_task_count == len(task_results)
        aggregate_score_sum = float(sum(t.score for t in task_results))
        aggregate_score_mean = aggregate_score_sum / float(len(task_results))
        aggregate_min_score = float(min(t.score for t in task_results))
        aggregate_lift_sum = float(sum(float(t.metrics.get("cube_lift", 0.0)) for t in task_results))

        task_score_json = {t.keyframe: t.score for t in task_results}
        task_feasible_json = {t.keyframe: bool(t.feasible) for t in task_results}
        task_choice_json = {t.keyframe: t.chosen_label for t in task_results}
        task_ctrl_json = {t.keyframe: t.chosen_ctrl.tolist() for t in task_results}

        row = {
            "candidate_id": idx,
            "scene_xml": str(scene_xml),
            "all_tasks_feasible": all_tasks_feasible,
            "feasible_task_count": feasible_task_count,
            "aggregate_score_sum": aggregate_score_sum,
            "aggregate_score_mean": aggregate_score_mean,
            "aggregate_min_score": aggregate_min_score,
            "aggregate_lift_sum": aggregate_lift_sum,
            "thumb_x": float(morphology.thumb_x),
            "thumb_y": float(morphology.thumb_y),
            "thumb_len": float(morphology.thumb_len),
            "index_x": float(morphology.index_x),
            "index_y": float(morphology.index_y),
            "index_len": float(morphology.index_len),
            "middle_x": float(morphology.middle_x),
            "middle_y": float(morphology.middle_y),
            "middle_len": float(morphology.middle_len),
            "morph_distance_from_base": float(_morph_distance(morphology, base_morph)),
            "task_score_json": json.dumps(task_score_json),
            "task_feasible_json": json.dumps(task_feasible_json),
            "task_choice_json": json.dumps(task_choice_json),
            "task_ctrl_json": json.dumps(task_ctrl_json),
        }
        rows.append(row)

        if idx % 50 == 0 or idx == len(candidates) - 1:
            print(
                f"[sample {idx+1}/{len(candidates)}] feasible_task_count={feasible_task_count} "
                f"all_tasks_feasible={all_tasks_feasible} agg_mean={aggregate_score_mean:.4f}"
            )

    # Multiobjective ranking: prioritize all-task feasibility, then mean score, then worst-task score.
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            int(bool(r["all_tasks_feasible"])),
            int(r["feasible_task_count"]),
            float(r["aggregate_score_mean"]),
            float(r["aggregate_min_score"]),
            float(r["aggregate_lift_sum"]),
        ),
        reverse=True,
    )
    top_rows = rows_sorted[: max(0, args.top_k_gifs)]

    gifs_dir = out_dir / "top5_gifs"
    gifs_dir.mkdir(parents=True, exist_ok=True)
    gif_manifest: list[dict[str, object]] = []

    for rank, r in enumerate(top_rows, start=1):
        scene_xml = Path(str(r["scene_xml"]))
        ctrl_map = json.loads(str(r["task_ctrl_json"]))
        task_gifs: dict[str, str] = {}
        for keyframe in args.keyframes:
            ctrl = np.asarray(ctrl_map[keyframe], dtype=np.float64)
            evaluator = Phase1GraspEvaluator(scene_xml=scene_xml, keyframe=keyframe, cfg=eval_cfg)
            gif_path = gifs_dir / f"rank{rank:02d}_{keyframe}.gif"
            evaluator.render_rollout_gif(
                ctrl,
                gif_path,
                width=args.gif_width,
                height=args.gif_height,
                fps=args.gif_fps,
                frame_stride=args.gif_frame_stride,
            )
            task_gifs[keyframe] = str(gif_path)
        gif_manifest.append(
            {
                "rank": rank,
                "candidate_id": int(r["candidate_id"]),
                "scene_xml": str(scene_xml),
                "all_tasks_feasible": bool(r["all_tasks_feasible"]),
                "feasible_task_count": int(r["feasible_task_count"]),
                "aggregate_score_mean": float(r["aggregate_score_mean"]),
                "aggregate_min_score": float(r["aggregate_min_score"]),
                "task_gifs": task_gifs,
            }
        )

    rolling_rows = _rolling_efficiency(rows, args.window)
    _plot_rolling_efficiency(rolling_rows, out_dir)

    _write_csv(rows, out_dir / "all_candidates_multitask.csv")
    _write_csv(task_rows, out_dir / "all_task_results.csv")
    _write_csv(top_rows, out_dir / "top5_candidates.csv")
    _write_csv(rolling_rows, out_dir / "rolling_efficiency.csv")
    (out_dir / "top5_gifs" / "gif_manifest.json").write_text(json.dumps(gif_manifest, indent=2), encoding="utf-8")

    summary = {
        "tag": tag,
        "scene_xml": str(args.scene_xml),
        "keyframes": args.keyframes,
        "samples": args.samples,
        "seed": args.seed,
        "window": args.window,
        "fp_refresh_interval": args.fp_refresh_interval,
        "morph_sort": args.morph_sort,
        "bounds": asdict(bounds),
        "perturb": {
            "x": args.x_perturb,
            "y": args.y_perturb,
            "len": args.len_perturb,
        },
        "feasibility": {
            "max_mean_tip_distance": args.max_mean_tip_distance,
            "min_contacts": args.min_contacts,
        },
        "interval_adaptation": {
            "count": interval_adapt_count,
            "seconds_total": interval_adapt_seconds,
            "cfg": asdict(interval_cfg),
        },
        "sparse_adaptation": {
            "seconds_total": sparse_adapt_seconds,
            "cfg": asdict(sparse_cfg),
        },
        "task_eval_seconds": task_eval_seconds,
        "overall": {
            "all_task_feasible_count": int(sum(1 for r in rows if bool(r["all_tasks_feasible"]))),
            "mean_feasible_tasks_per_sample": float(np.mean([int(r["feasible_task_count"]) for r in rows])) if rows else 0.0,
            "mean_aggregate_score": float(np.mean([float(r["aggregate_score_mean"]) for r in rows])) if rows else 0.0,
            "max_aggregate_score": float(np.max([float(r["aggregate_score_mean"]) for r in rows])) if rows else 0.0,
        },
        "top5": [
            {
                "rank": i + 1,
                "candidate_id": int(r["candidate_id"]),
                "all_tasks_feasible": bool(r["all_tasks_feasible"]),
                "feasible_task_count": int(r["feasible_task_count"]),
                "aggregate_score_mean": float(r["aggregate_score_mean"]),
                "aggregate_min_score": float(r["aggregate_min_score"]),
                "scene_xml": str(r["scene_xml"]),
            }
            for i, r in enumerate(top_rows)
        ],
        "wall_time_seconds": float(time.perf_counter() - run_t0),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Run6 combined complete: {out_dir}")


if __name__ == "__main__":
    main()
