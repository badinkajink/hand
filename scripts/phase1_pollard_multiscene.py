from __future__ import annotations
# pyright: reportMissingImports=false

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

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
from morphohand.sampling import (  # noqa: E402
    FeasibilityCriteria,
    FoundationalPose,
    MorphologyBounds,
    adapt_foundational_ctrl,
    add_foundational_keyframe,
    is_feasible,
    load_base_morphology,
    load_best_pose_with_prefix,
    load_foundational_poses_with_scene,
    morph_distance,
    morph_row_fields,
    morph_suffix,
    pareto_front_indices,
    plot_feasible_scatter,
    sample_morphologies,
    simulate_settle_qpos,
    write_csv,
    write_rigid_scene_with_object_size,
)


@dataclass(frozen=True)
class SceneSpec:
    scene_key: str
    display_name: str
    size_x: float
    size_y: float
    size_z: float
    foundational_poses: list[FoundationalPose]
    criteria: FeasibilityCriteria


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Large-scale Pollard-style morphology sampling across cube + prism scenes with "
            "foundational-pose feasibility checks, keyframe export, and top-k GIF rendering."
        )
    )
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=["mujoco", "mjwarp", "comfree-warp"],
        default="mujoco",
        help="Physics backend used by evaluator rollouts.",
    )
    parser.add_argument("--comfree-stiffness", type=float, default=0.2)
    parser.add_argument("--comfree-damping", type=float, default=0.001)
    parser.add_argument("--backend-nworld", type=int, default=1)
    parser.add_argument("--backend-nconmax", type=int, default=200)
    parser.add_argument("--backend-njmax", type=int, default=2000)
    parser.add_argument("--backend-sync-interval", type=int, default=1)
    parser.add_argument("--metric-sample-interval", type=int, default=1)
    parser.add_argument(
        "--speed-mode",
        choices=["accurate", "balanced", "aggressive"],
        default="accurate",
    )
    parser.add_argument(
        "--metric-collection-mode",
        choices=["sampled", "terminal"],
        default="sampled",
    )
    parser.add_argument("--x-perturb", type=float, default=0.012)
    parser.add_argument("--y-perturb", type=float, default=0.012)
    parser.add_argument("--len-perturb", type=float, default=0.012)
    parser.add_argument("--x-min", type=float, default=-0.03)
    parser.add_argument("--x-max", type=float, default=0.03)
    parser.add_argument("--y-min", type=float, default=-0.03)
    parser.add_argument("--y-max", type=float, default=0.03)
    parser.add_argument("--len-min", type=float, default=0.0)
    parser.add_argument("--len-max", type=float, default=0.035)
    parser.add_argument("--cube-max-mean-tip-distance", type=float, default=0.012)
    parser.add_argument("--cube-min-contacts", type=float, default=2.0)
    parser.add_argument("--cube-min-finger-contact-persistence", type=float, default=0.55)
    parser.add_argument("--cube-max-finger-yaw-drift", type=float, default=0.30)
    parser.add_argument("--prism-max-mean-tip-distance", type=float, default=0.03)
    parser.add_argument("--prism-min-contacts", type=float, default=1.0)
    parser.add_argument("--prism-min-finger-contact-persistence", type=float, default=0.45)
    parser.add_argument("--prism-max-finger-yaw-drift", type=float, default=0.40)
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
    parser.add_argument("--top-k-gifs", type=int, default=5)
    parser.add_argument("--refine-top-k", action="store_true")
    parser.add_argument("--refine-pool-size", type=int, default=15)
    parser.add_argument("--refine-iterations", type=int, default=8)
    parser.add_argument("--refine-population", type=int, default=24)
    parser.add_argument("--refine-elite-fraction", type=float, default=0.25)
    parser.add_argument("--refine-sigma-init", type=float, default=0.08)
    parser.add_argument("--refine-seed", type=int, default=0)
    parser.add_argument(
        "--fp-adaptation",
        choices=["none", "interval-open", "interval-initial-fp", "sparse-per-morph", "local-perturbation"],
        default="none",
    )
    parser.add_argument(
        "--morph-sort",
        choices=["none", "distance"],
        default="none",
    )
    parser.add_argument("--fp-refresh-interval", type=int, default=40)
    parser.add_argument("--fp-adapt-iterations", type=int, default=12)
    parser.add_argument("--fp-adapt-population", type=int, default=24)
    parser.add_argument("--fp-adapt-elite-fraction", type=float, default=0.25)
    parser.add_argument("--fp-adapt-sigma-init", type=float, default=0.08)
    parser.add_argument("--fp-adapt-seed", type=int, default=0)
    parser.add_argument(
        "--base-scene-xml",
        type=Path,
        default=PROJECT_ROOT / "assets" / "mjcf" / "scene.xml",
    )
    parser.add_argument(
        "--cube-foundational-run-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase1" / "run_20260410_163959",
    )
    parser.add_argument(
        "--prism-foundational-run-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase1" / "run_20260413_prism_y_sweep_mjx_autodiff",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase1",
    )
    parser.add_argument("--tag", default=None)
    return parser


def _build_scene_specs(args: argparse.Namespace):
    cube_poses, cube_scene = load_foundational_poses_with_scene(args.cube_foundational_run_dir)
    base_morphology = load_base_morphology(cube_scene, keyframe_name="open")

    prism25 = load_best_pose_with_prefix(args.prism_foundational_run_dir, "y0d0250", "prism_y0d0250")
    prism30 = load_best_pose_with_prefix(args.prism_foundational_run_dir, "y0d0300", "prism_y0d0300")
    prism35 = load_best_pose_with_prefix(args.prism_foundational_run_dir, "y0d0350", "prism_y0d0350")

    cube_criteria = FeasibilityCriteria(
        max_mean_tip_distance=args.cube_max_mean_tip_distance,
        min_contacts=args.cube_min_contacts,
        min_finger_contact_persistence=args.cube_min_finger_contact_persistence,
        max_finger_yaw_drift=args.cube_max_finger_yaw_drift,
    )
    prism_criteria = FeasibilityCriteria(
        max_mean_tip_distance=args.prism_max_mean_tip_distance,
        min_contacts=args.prism_min_contacts,
        min_finger_contact_persistence=args.prism_min_finger_contact_persistence,
        max_finger_yaw_drift=args.prism_max_finger_yaw_drift,
    )

    specs = [
        SceneSpec("cube", "cube", 0.02, 0.02, 0.02, cube_poses, cube_criteria),
        SceneSpec("prism1", "prism_y0d0250", 0.02, 0.025, 0.02, [prism25], prism_criteria),
        SceneSpec("prism2", "prism_y0d0300", 0.02, 0.03, 0.02, [prism30], prism_criteria),
        SceneSpec("prism3", "prism_y0d0350", 0.02, 0.035, 0.02, [prism35], prism_criteria),
    ]
    return specs, base_morphology


def _make_evaluator(args: argparse.Namespace, scene_xml: Path, eval_cfg: Phase1EvalConfig) -> Phase1GraspEvaluator:
    return Phase1GraspEvaluator(
        scene_xml=scene_xml,
        keyframe="open",
        cfg=eval_cfg,
        backend=args.backend,
        comfree_stiffness=args.comfree_stiffness,
        comfree_damping=args.comfree_damping,
        backend_nworld=args.backend_nworld,
        backend_nconmax=args.backend_nconmax,
        backend_njmax=args.backend_njmax,
        backend_sync_interval=args.backend_sync_interval,
        metric_sample_interval=args.metric_sample_interval,
        speed_mode=args.speed_mode,
        metric_collection_mode=args.metric_collection_mode,
    )


_METRIC_KEYS: tuple[str, ...] = (
    "cube_lift", "cube_tip_contacts", "mean_tip_distance", "cube_vel_norm",
    "cube_xy_drift", "cube_z_drop_from_peak", "contact_persistence",
    "thumb_contact_persistence", "index_contact_persistence", "middle_contact_persistence",
    "all_finger_contact_persistence", "min_finger_contact_persistence",
    "finger_persistence_imbalance", "finger_yaw_drift", "finger_flex_drift",
)


def _metric_fields(metrics: dict[str, float]) -> dict[str, float]:
    return {k: float(metrics.get(k, 0.0)) for k in _METRIC_KEYS}


def _evaluate_pool(
    evaluator: Phase1GraspEvaluator,
    pool: list[tuple[str, np.ndarray]],
    criteria: FeasibilityCriteria,
) -> list[tuple[str, np.ndarray, float, dict[str, float], bool]]:
    out: list[tuple[str, np.ndarray, float, dict[str, float], bool]] = []
    for label, ctrl in pool:
        score, metrics = evaluator.evaluate(ctrl)
        out.append((label, ctrl, score, metrics, is_feasible(metrics, criteria)))
    return out


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    tag = args.tag or datetime.now().strftime("run_%Y%m%d_%H%M%S_pollard_multiscene")
    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    specs, base_morphology = _build_scene_specs(args)

    bounds = MorphologyBounds(
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        len_min=args.len_min, len_max=args.len_max,
    )
    candidates = sample_morphologies(
        base=base_morphology,
        sample_count=args.samples,
        rng=rng,
        bounds=bounds,
        x_perturb=args.x_perturb,
        y_perturb=args.y_perturb,
        len_perturb=args.len_perturb,
    )
    if args.morph_sort == "distance":
        candidates.sort(key=lambda m: morph_distance(m, base_morphology))
        print(
            f"[morph-sort] {len(candidates)} candidates by distance from base "
            f"(min={morph_distance(candidates[0], base_morphology):.6f}, "
            f"max={morph_distance(candidates[-1], base_morphology):.6f})"
        )

    eval_cfg = Phase1EvalConfig(
        settle_steps=args.settle_steps,
        lift_steps=args.lift_steps,
        hold_steps=args.hold_steps,
        lift_delta_z=args.lift_delta_z,
        lift_ramp_steps=args.lift_ramp_steps,
        objective_weight_min_finger_persistence=args.objective_weight_min_finger_persistence,
        objective_weight_finger_persistence_imbalance_penalty=(
            args.objective_weight_finger_persistence_imbalance_penalty
        ),
        objective_weight_finger_yaw_drift_penalty=args.objective_weight_finger_yaw_drift_penalty,
        objective_weight_finger_flex_drift_penalty=args.objective_weight_finger_flex_drift_penalty,
    )
    fp_adapt_cfg = Phase1OptimizationConfig(
        iterations=args.fp_adapt_iterations,
        population=args.fp_adapt_population,
        elite_fraction=args.fp_adapt_elite_fraction,
        sigma_init=args.fp_adapt_sigma_init,
        seed=args.fp_adapt_seed,
    )

    global_rows: list[dict[str, object]] = []
    scene_summaries: dict[str, dict[str, object]] = {}

    for spec in specs:
        scene_dir = out_dir / spec.scene_key
        generated_dir = scene_dir / "generated_mjcf"
        generated_dir.mkdir(parents=True, exist_ok=True)

        all_rows: list[dict[str, object]] = []
        feasible_rows: list[dict[str, object]] = []
        scene_eval_seconds_total = 0.0
        scene_adapt_seconds_total = 0.0
        scene_adapt_count = 0

        default_pose = max(spec.foundational_poses, key=lambda p: p.score)
        baseline_fp_ctrl = np.asarray(default_pose.finger_ctrl, dtype=np.float64)
        interval_fp_ctrl = baseline_fp_ctrl.copy()

        for idx, morphology in enumerate(candidates):
            scene_xml = generated_dir / f"scene_{spec.scene_key}_{morph_suffix(morphology)}.xml"
            write_rigid_scene_with_object_size(
                base_scene_xml=args.base_scene_xml,
                output_scene_xml=scene_xml,
                morphology=morphology,
                object_body_name="cube",
                object_geom_type="box",
                size_xyz=(spec.size_x, spec.size_y, spec.size_z),
            )

            evaluator = _make_evaluator(args, scene_xml, eval_cfg)

            ctrl_adapted, triggered, adapt_secs, adapt_source = adapt_foundational_ctrl(
                mode=args.fp_adaptation,
                candidate_idx=idx,
                interval_ctrl=interval_fp_ctrl,
                evaluator=evaluator,
                adapt_cfg=fp_adapt_cfg,
                refresh_interval=args.fp_refresh_interval,
            )
            if args.fp_adaptation in {"interval-open", "interval-initial-fp"} and triggered:
                interval_fp_ctrl = ctrl_adapted.copy()
            scene_adapt_seconds_total += adapt_secs
            scene_adapt_count += int(triggered)

            t0 = time.perf_counter()
            if args.fp_adaptation == "none":
                pool = [(p.label, np.asarray(p.finger_ctrl, dtype=np.float64)) for p in spec.foundational_poses]
            else:
                pool = [(f"adapted_{args.fp_adaptation}", ctrl_adapted)]
                pool.extend(
                    (p.label, np.asarray(p.finger_ctrl, dtype=np.float64)) for p in spec.foundational_poses
                )
            evals = _evaluate_pool(evaluator, pool, spec.criteria)
            eval_seconds = float(time.perf_counter() - t0)
            scene_eval_seconds_total += eval_seconds

            feasible_hits = [e for e in evals if e[4]]
            feasible_pose_count = len(feasible_hits)
            any_feasible = feasible_pose_count > 0
            chosen_label, chosen_ctrl, chosen_score, chosen_metrics, _ = max(
                feasible_hits if feasible_hits else evals, key=lambda e: e[2]
            )

            qpos_foundational, ctrl_foundational = simulate_settle_qpos(
                scene_xml=scene_xml,
                keyframe_name="open",
                finger_ctrl=chosen_ctrl,
                settle_steps=args.settle_steps,
            )
            add_foundational_keyframe(
                scene_xml=scene_xml,
                key_name="foundational",
                qpos=qpos_foundational,
                ctrl=ctrl_foundational,
            )

            row: dict[str, object] = {
                "scene_key": spec.scene_key,
                "candidate_id": str(idx),
                "scene_xml": str(scene_xml),
                "selected_foundational_pose": chosen_label,
                "feasible": str(any_feasible),
                "backend": args.backend,
                "fp_adaptation": args.fp_adaptation,
                "fp_adapt_triggered": str(triggered),
                "fp_adapt_source": adapt_source,
                "fp_adapt_seconds": float(adapt_secs),
                "evaluation_seconds": float(eval_seconds),
                "feasibility_max_mean_tip_distance": float(spec.criteria.max_mean_tip_distance),
                "feasibility_min_contacts": float(spec.criteria.min_contacts),
                "feasibility_min_finger_contact_persistence": float(
                    spec.criteria.min_finger_contact_persistence or 0.0
                ),
                "feasibility_max_finger_yaw_drift": float(
                    spec.criteria.max_finger_yaw_drift or 0.0
                ),
                "feasible_pose_count": float(feasible_pose_count),
                "foundational_pose_count": float(len(spec.foundational_poses)),
                "score": float(chosen_score),
                **_metric_fields(chosen_metrics),
                **morph_row_fields(morphology),
                "morph_distance_from_base": float(morph_distance(morphology, base_morphology)),
                "chosen_ctrl_json": json.dumps(chosen_ctrl.tolist()),
            }
            all_rows.append(row)
            global_rows.append(row)
            if any_feasible:
                feasible_rows.append(row)

        pareto_idx = pareto_front_indices(feasible_rows)
        pareto_rows = [feasible_rows[i] for i in pareto_idx]

        write_csv(all_rows, scene_dir / "all_candidates.csv")
        write_csv(feasible_rows, scene_dir / "feasible_candidates.csv")
        write_csv(pareto_rows, scene_dir / "pareto_front.csv")
        plot_feasible_scatter(feasible_rows, pareto_idx, scene_dir, title=f"{spec.display_name}: Feasible Morphologies")

        ranking_pool = feasible_rows if feasible_rows else all_rows

        if args.refine_top_k and ranking_pool:
            refine_cfg = Phase1OptimizationConfig(
                iterations=args.refine_iterations,
                population=args.refine_population,
                elite_fraction=args.refine_elite_fraction,
                sigma_init=args.refine_sigma_init,
                seed=args.refine_seed,
            )
            candidate_pool = sorted(ranking_pool, key=lambda r: float(r["score"]), reverse=True)[
                : max(args.top_k_gifs, args.refine_pool_size)
            ]
            refined_rows: list[dict[str, object]] = []
            for row in candidate_pool:
                evaluator = _make_evaluator(args, Path(str(row["scene_xml"])), eval_cfg)
                init_ctrl = np.asarray(json.loads(str(row["chosen_ctrl_json"])), dtype=np.float64)
                refined = optimize_finger_controls(
                    evaluator=evaluator, cfg=refine_cfg, initial_finger_ctrl=init_ctrl
                )
                ctrl_refined = np.asarray(refined["best_finger_ctrl"], dtype=np.float64)
                score_refined, metrics_refined = evaluator.evaluate(ctrl_refined)

                updated = dict(row)
                updated["score"] = float(score_refined)
                updated.update(_metric_fields(metrics_refined))
                updated["chosen_ctrl_json"] = json.dumps(ctrl_refined.tolist())
                updated["refined_for_topk"] = "True"
                refined_rows.append(updated)

            top_rows = sorted(refined_rows, key=lambda r: float(r["score"]), reverse=True)[: args.top_k_gifs]
        else:
            top_rows = sorted(ranking_pool, key=lambda r: float(r["score"]), reverse=True)[: args.top_k_gifs]

        video_dir = scene_dir / "top_videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        top_rows_export: list[dict[str, object]] = []
        for rank, row in enumerate(top_rows, start=1):
            evaluator = _make_evaluator(args, Path(str(row["scene_xml"])), eval_cfg)
            ctrl = np.asarray(json.loads(str(row["chosen_ctrl_json"])), dtype=np.float64)
            video_path = video_dir / f"rank{rank:02d}_candidate{int(row['candidate_id']):04d}.mp4"
            evaluator.render_rollout(ctrl, video_path)
            item = dict(row)
            item["rank"] = float(rank)
            item["video_path"] = str(video_path)
            if "refined_for_topk" not in item:
                item["refined_for_topk"] = "False"
            top_rows_export.append(item)

        write_csv(top_rows_export, scene_dir / "top5_with_videos.csv")

        scene_summaries[spec.scene_key] = {
            "total_candidates": len(all_rows),
            "feasible_candidates": len(feasible_rows),
            "pareto_size": len(pareto_rows),
            "video_count": len(top_rows_export),
            "video_ranking_source": "feasible" if feasible_rows else "all_candidates",
            "backend": args.backend,
            "fp_adaptation": args.fp_adaptation,
            "fp_adapt_count": scene_adapt_count,
            "fp_adapt_seconds_total": scene_adapt_seconds_total,
            "evaluation_seconds_total": scene_eval_seconds_total,
            "mean_evaluation_seconds": scene_eval_seconds_total / max(1, len(all_rows)),
        }

        print(
            f"[{spec.scene_key}] total={len(all_rows)} feasible={len(feasible_rows)} "
            f"pareto={len(pareto_rows)} videos={len(top_rows_export)}"
        )

    write_csv(global_rows, out_dir / "all_scenes_candidates.csv")

    summary = {
        "tag": tag,
        "samples": args.samples,
        "seed": args.seed,
        "backend": args.backend,
        "speed_mode": args.speed_mode,
        "metric_collection_mode": args.metric_collection_mode,
        "fp_adaptation": args.fp_adaptation,
        "fp_refresh_interval": args.fp_refresh_interval,
        "morph_sort": args.morph_sort,
        "bounds": asdict(bounds),
        "fp_adapt_config": asdict(fp_adapt_cfg),
        "backend_config": {
            "comfree_stiffness": args.comfree_stiffness,
            "comfree_damping": args.comfree_damping,
            "backend_nworld": args.backend_nworld,
            "backend_nconmax": args.backend_nconmax,
            "backend_njmax": args.backend_njmax,
            "backend_sync_interval": args.backend_sync_interval,
            "metric_sample_interval": args.metric_sample_interval,
            "speed_mode": args.speed_mode,
            "metric_collection_mode": args.metric_collection_mode,
        },
        "scene_summaries": scene_summaries,
        "paths": {
            "out_dir": str(out_dir),
            "cube_foundational_run_dir": str(args.cube_foundational_run_dir),
            "prism_foundational_run_dir": str(args.prism_foundational_run_dir),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Multi-scene Pollard run complete: {out_dir}")


if __name__ == "__main__":
    main()
