from __future__ import annotations
# pyright: reportMissingImports=false

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from morphohand.optimization.contact_targets import ContactTargetSet  # noqa: E402
from morphohand.optimization.phase1_grasp import (  # noqa: E402
    Phase1EvalConfig,
    Phase1GraspEvaluator,
    Phase1OptimizationConfig,
    optimize_finger_controls,
)
from morphohand.sampling import (  # noqa: E402
    FeasibilityCriteria,
    FoundationalPose,
    MorphologyBounds,
    is_feasible,
    load_foundational_poses,
    morph_distance,
    morph_row_fields,
    morph_suffix,
    parse_morphology_from_keyframe,
    sample_morphologies,
    write_csv,
    write_rigid_scene_with_object_size,
)


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


def _criteria_for_keyframe(args: argparse.Namespace, keyframe: str) -> FeasibilityCriteria:
    is_vertical = keyframe in set(args.vertical_keyframes)
    max_xy = args.vertical_max_cube_xy_drift if is_vertical else args.max_cube_xy_drift
    max_yaw = args.vertical_max_cube_yaw_drift if is_vertical else args.max_cube_yaw_drift
    max_tilt = args.vertical_max_cube_axis_tilt if is_vertical else args.max_cube_axis_tilt
    max_ang = args.vertical_max_cube_ang_drift if is_vertical else args.max_cube_ang_drift
    return FeasibilityCriteria(
        max_mean_tip_distance=args.max_mean_tip_distance,
        min_contacts=args.min_contacts,
        max_cube_xy_drift=max_xy,
        max_cube_yaw_drift=max_yaw,
        max_cube_axis_tilt=max_tilt,
        max_cube_ang_drift=max_ang,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run6 combined multitask sampler: evaluate 3 screwdriver keyframes per morphology, "
            "combine sparse+interval adaptation in one run, track rolling efficiency, render top-5 gifs."
        )
    )
    parser.add_argument(
        "--scene-xml",
        type=Path,
        default=PROJECT_ROOT / "assets" / "mjcf" / "scene_screwdriver_medium.xml",
    )
    parser.add_argument("--keyframes", nargs="+", default=["open_flat", "open_vertical", "open_90vertical"])
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
    parser.add_argument("--max-cube-xy-drift", type=float, default=0.018)
    parser.add_argument("--max-cube-yaw-drift", type=float, default=0.30)
    parser.add_argument("--max-cube-axis-tilt", type=float, default=0.22)
    parser.add_argument("--max-cube-ang-drift", type=float, default=0.50)
    parser.add_argument("--vertical-keyframes", nargs="+", default=["open_vertical", "open_90vertical"])
    parser.add_argument("--vertical-max-cube-xy-drift", type=float, default=0.010)
    parser.add_argument("--vertical-max-cube-yaw-drift", type=float, default=0.12)
    parser.add_argument("--vertical-max-cube-axis-tilt", type=float, default=0.10)
    parser.add_argument("--vertical-max-cube-ang-drift", type=float, default=0.22)
    parser.add_argument("--settle-steps", type=int, default=240)
    parser.add_argument("--lift-steps", type=int, default=220)
    parser.add_argument("--hold-steps", type=int, default=140)
    parser.add_argument("--lift-delta-z", type=float, default=0.05)
    parser.add_argument("--lift-ramp-steps", type=int, default=100)
    parser.add_argument("--pivot-steps", type=int, default=0)
    parser.add_argument("--pivot-ramp-steps", type=int, default=80)
    parser.add_argument("--pivot-delta-rx", type=float, default=0.0)
    parser.add_argument("--pivot-delta-ry", type=float, default=0.0)
    parser.add_argument("--pivot-delta-rz", type=float, default=0.0)
    parser.add_argument("--objective-weight-min-finger-persistence", type=float, default=2.4)
    parser.add_argument(
        "--objective-weight-finger-persistence-imbalance-penalty", type=float, default=1.2,
    )
    parser.add_argument("--objective-weight-finger-yaw-drift-penalty", type=float, default=1.0)
    parser.add_argument("--objective-weight-finger-flex-drift-penalty", type=float, default=0.5)
    parser.add_argument("--objective-weight-cube-yaw-drift-penalty", type=float, default=4.0)
    parser.add_argument("--objective-weight-cube-axis-tilt-penalty", type=float, default=6.0)
    parser.add_argument("--objective-weight-cube-ang-drift-penalty", type=float, default=2.0)
    parser.add_argument(
        "--contact-targets-yaml",
        type=Path,
        default=None,
        help="Optional contact-patch YAML; enables contact_map-style targeting on every evaluator built in the sweep.",
    )
    parser.add_argument(
        "--objective-weight-contact-target-reward",
        type=float,
        default=0.0,
        help="Reward weight for hitting contact-target patches (only used when --contact-targets-yaml is set).",
    )
    parser.add_argument(
        "--objective-weight-contact-target-distance-penalty",
        type=float,
        default=0.0,
        help="Penalty weight for mean tip-to-patch distance (only used when --contact-targets-yaml is set).",
    )
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


def _rolling_efficiency(rows: list[dict[str, object]], window: int) -> list[dict[str, float]]:
    n = len(rows)
    if n == 0:
        return []
    total_tasks = float(len(json.loads(str(rows[0]["task_score_json"]))))
    out: list[dict[str, float]] = []
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
        [(r["window_start"] + r["window_end"]) * 0.5 for r in rolling_rows], dtype=np.float64
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
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        len_min=args.len_min, len_max=args.len_max,
    )
    criteria_by_keyframe: dict[str, FeasibilityCriteria] = {
        keyframe: _criteria_for_keyframe(args, keyframe) for keyframe in args.keyframes
    }

    eval_cfg = Phase1EvalConfig(
        settle_steps=args.settle_steps,
        lift_steps=args.lift_steps,
        hold_steps=args.hold_steps,
        lift_delta_z=args.lift_delta_z,
        lift_ramp_steps=args.lift_ramp_steps,
        pivot_steps=args.pivot_steps,
        pivot_ramp_steps=args.pivot_ramp_steps,
        pivot_delta_rx=args.pivot_delta_rx,
        pivot_delta_ry=args.pivot_delta_ry,
        pivot_delta_rz=args.pivot_delta_rz,
        objective_weight_min_finger_persistence=args.objective_weight_min_finger_persistence,
        objective_weight_finger_persistence_imbalance_penalty=args.objective_weight_finger_persistence_imbalance_penalty,
        objective_weight_finger_yaw_drift_penalty=args.objective_weight_finger_yaw_drift_penalty,
        objective_weight_finger_flex_drift_penalty=args.objective_weight_finger_flex_drift_penalty,
        objective_weight_cube_yaw_drift_penalty=args.objective_weight_cube_yaw_drift_penalty,
        objective_weight_cube_axis_tilt_penalty=args.objective_weight_cube_axis_tilt_penalty,
        objective_weight_cube_ang_drift_penalty=args.objective_weight_cube_ang_drift_penalty,
        objective_weight_contact_target_reward=args.objective_weight_contact_target_reward,
        objective_weight_contact_target_distance_penalty=args.objective_weight_contact_target_distance_penalty,
    )
    contact_target_set = (
        ContactTargetSet.from_yaml(args.contact_targets_yaml)
        if args.contact_targets_yaml is not None
        else None
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
    base_morph = parse_morphology_from_keyframe(args.scene_xml, keyframe_name=args.keyframes[0])

    for keyframe in args.keyframes:
        poses = load_foundational_poses(args.foundational_root / keyframe, keyframe_name=keyframe)
        foundational_by_keyframe[keyframe] = poses
        interval_ctrl_by_keyframe[keyframe] = np.asarray(poses[0].finger_ctrl, dtype=np.float64).copy()

    candidates = sample_morphologies(
        base=base_morph,
        sample_count=args.samples,
        rng=rng,
        bounds=bounds,
        x_perturb=args.x_perturb,
        y_perturb=args.y_perturb,
        len_perturb=args.len_perturb,
    )
    if args.morph_sort == "distance":
        candidates.sort(key=lambda m: morph_distance(m, base_morph))

    gen_dir = out_dir / "generated_mjcf"
    gen_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []

    interval_adapt_count = {k: 0 for k in args.keyframes}
    interval_adapt_seconds = {k: 0.0 for k in args.keyframes}
    sparse_adapt_seconds = {k: 0.0 for k in args.keyframes}
    task_eval_seconds = {k: 0.0 for k in args.keyframes}

    for idx, morphology in enumerate(candidates):
        scene_xml = gen_dir / f"scene_multi_{morph_suffix(morphology)}.xml"
        write_rigid_scene_with_object_size(
            base_scene_xml=args.scene_xml,
            output_scene_xml=scene_xml,
            morphology=morphology,
            size_xyz=None,
        )

        task_results: list[TaskResult] = []

        for keyframe in args.keyframes:
            evaluator = Phase1GraspEvaluator(
                scene_xml=scene_xml,
                keyframe=keyframe,
                cfg=eval_cfg,
                contact_target_set=contact_target_set,
            )
            interval_ctrl = interval_ctrl_by_keyframe[keyframe]
            criteria = criteria_by_keyframe[keyframe]

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
            evals.append(("interval", interval_ctrl, s_interval, m_interval, is_feasible(m_interval, criteria)))

            s_sparse, m_sparse = evaluator.evaluate(sparse_ctrl)
            evals.append(("sparse", sparse_ctrl, s_sparse, m_sparse, is_feasible(m_sparse, criteria)))

            for pose in foundational_by_keyframe[keyframe]:
                s_pose, m_pose = evaluator.evaluate(pose.finger_ctrl)
                evals.append(
                    (
                        pose.label,
                        np.asarray(pose.finger_ctrl, dtype=np.float64),
                        s_pose,
                        m_pose,
                        is_feasible(m_pose, criteria),
                    )
                )

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
                    "cube_yaw_drift": float(task_result.metrics.get("cube_yaw_drift", 0.0)),
                    "cube_axis_tilt": float(task_result.metrics.get("cube_axis_tilt", 0.0)),
                    "cube_ang_drift": float(task_result.metrics.get("cube_ang_drift", 0.0)),
                    "finger_flex_drift": float(task_result.metrics.get("finger_flex_drift", 0.0)),
                    "interval_triggered": task_result.interval_triggered,
                    "interval_adapt_seconds": task_result.interval_adapt_seconds,
                    "sparse_adapt_seconds": task_result.sparse_adapt_seconds,
                }
            )

        feasible_task_count = int(sum(1 for t in task_results if t.feasible))
        all_tasks_feasible = feasible_task_count == len(task_results)
        scores = [t.score for t in task_results]
        aggregate_score_sum = float(sum(scores))
        aggregate_score_mean = aggregate_score_sum / float(len(task_results))
        aggregate_min_score = float(min(scores))
        aggregate_lift_sum = float(sum(float(t.metrics.get("cube_lift", 0.0)) for t in task_results))

        row: dict[str, object] = {
            "candidate_id": idx,
            "scene_xml": str(scene_xml),
            "all_tasks_feasible": all_tasks_feasible,
            "feasible_task_count": feasible_task_count,
            "aggregate_score_sum": aggregate_score_sum,
            "aggregate_score_mean": aggregate_score_mean,
            "aggregate_min_score": aggregate_min_score,
            "aggregate_lift_sum": aggregate_lift_sum,
            **morph_row_fields(morphology),
            "morph_distance_from_base": float(morph_distance(morphology, base_morph)),
            "task_score_json": json.dumps({t.keyframe: t.score for t in task_results}),
            "task_feasible_json": json.dumps({t.keyframe: bool(t.feasible) for t in task_results}),
            "task_choice_json": json.dumps({t.keyframe: t.chosen_label for t in task_results}),
            "task_ctrl_json": json.dumps({t.keyframe: t.chosen_ctrl.tolist() for t in task_results}),
        }
        rows.append(row)

        if idx % 50 == 0 or idx == len(candidates) - 1:
            print(
                f"[sample {idx+1}/{len(candidates)}] feasible_task_count={feasible_task_count} "
                f"all_tasks_feasible={all_tasks_feasible} agg_mean={aggregate_score_mean:.4f}"
            )

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

    videos_dir = out_dir / "top5_videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    video_manifest: list[dict[str, object]] = []

    for rank, r in enumerate(top_rows, start=1):
        scene_xml = Path(str(r["scene_xml"]))
        ctrl_map = json.loads(str(r["task_ctrl_json"]))
        task_videos: dict[str, str] = {}
        for keyframe in args.keyframes:
            ctrl = np.asarray(ctrl_map[keyframe], dtype=np.float64)
            evaluator = Phase1GraspEvaluator(
                scene_xml=scene_xml,
                keyframe=keyframe,
                cfg=eval_cfg,
                contact_target_set=contact_target_set,
            )
            video_path = videos_dir / f"rank{rank:02d}_{keyframe}.mp4"
            evaluator.render_rollout(
                ctrl,
                video_path,
                width=args.gif_width,
                height=args.gif_height,
                fps=args.gif_fps,
                frame_stride=args.gif_frame_stride,
            )
            task_videos[keyframe] = str(video_path)
        video_manifest.append(
            {
                "rank": rank,
                "candidate_id": int(r["candidate_id"]),
                "scene_xml": str(scene_xml),
                "all_tasks_feasible": bool(r["all_tasks_feasible"]),
                "feasible_task_count": int(r["feasible_task_count"]),
                "aggregate_score_mean": float(r["aggregate_score_mean"]),
                "aggregate_min_score": float(r["aggregate_min_score"]),
                "task_videos": task_videos,
            }
        )

    rolling_rows = _rolling_efficiency(rows, args.window)
    _plot_rolling_efficiency(rolling_rows, out_dir)

    write_csv(rows, out_dir / "all_candidates_multitask.csv")
    write_csv(task_rows, out_dir / "all_task_results.csv")
    write_csv(top_rows, out_dir / "top5_candidates.csv")
    write_csv(rolling_rows, out_dir / "rolling_efficiency.csv")
    (out_dir / "top5_videos" / "video_manifest.json").write_text(
        json.dumps(video_manifest, indent=2), encoding="utf-8"
    )

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
        "perturb": {"x": args.x_perturb, "y": args.y_perturb, "len": args.len_perturb},
        "contact_targets": {
            "yaml": str(args.contact_targets_yaml) if args.contact_targets_yaml else None,
            "weight_reward": float(args.objective_weight_contact_target_reward),
            "weight_distance_penalty": float(args.objective_weight_contact_target_distance_penalty),
            "patch_count": (len(contact_target_set.patches) if contact_target_set else 0),
        },
        "feasibility": {
            "max_mean_tip_distance": args.max_mean_tip_distance,
            "min_contacts": args.min_contacts,
            "criteria_by_keyframe": {
                keyframe: asdict(criteria_by_keyframe[keyframe]) for keyframe in args.keyframes
            },
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
            "mean_feasible_tasks_per_sample": (
                float(np.mean([int(r["feasible_task_count"]) for r in rows])) if rows else 0.0
            ),
            "mean_aggregate_score": (
                float(np.mean([float(r["aggregate_score_mean"]) for r in rows])) if rows else 0.0
            ),
            "max_aggregate_score": (
                float(np.max([float(r["aggregate_score_mean"]) for r in rows])) if rows else 0.0
            ),
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
