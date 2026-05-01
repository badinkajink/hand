from __future__ import annotations

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
)
from morphohand.sampling import (  # noqa: E402
    FeasibilityCriteria,
    MorphologyBounds,
    adapt_foundational_ctrl,
    is_feasible,
    load_foundational_poses,
    morph_distance,
    morph_row_fields,
    morph_suffix,
    pareto_front_indices,
    parse_morphology_from_keyframe,
    plot_feasible_scatter,
    sample_morphologies,
    write_csv,
    write_rigid_scene_with_object_size,
)


@dataclass(frozen=True)
class KeyframeSpec:
    keyframe: str
    foundational_run_dir: Path
    max_mean_tip_distance: float
    min_contacts: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run6: morphology sampling on screwdriver medium scene across multiple keyframes "
            "with foundational-pose adaptation strategies."
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
        help="Contains one subdir per keyframe, each with seed subruns + summary.json.",
    )
    parser.add_argument("--samples", type=int, default=320)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fp-adaptation",
        choices=["none", "interval-initial-fp", "sparse-per-morph"],
        default="sparse-per-morph",
    )
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
        "--objective-weight-finger-persistence-imbalance-penalty", type=float, default=1.2,
    )
    parser.add_argument("--objective-weight-finger-yaw-drift-penalty", type=float, default=1.0)
    parser.add_argument("--objective-weight-finger-flex-drift-penalty", type=float, default=0.5)
    parser.add_argument("--adapt-iterations", type=int, default=12)
    parser.add_argument("--adapt-population", type=int, default=24)
    parser.add_argument("--adapt-elite-fraction", type=float, default=0.25)
    parser.add_argument("--adapt-sigma-init", type=float, default=0.08)
    parser.add_argument("--adapt-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "phase1")
    parser.add_argument("--tag", default=None)
    return parser


def _evaluate_pool(
    evaluator: Phase1GraspEvaluator,
    candidates: list[tuple[str, np.ndarray]],
    criteria: FeasibilityCriteria,
) -> list[tuple[str, np.ndarray, float, dict[str, float], bool]]:
    out: list[tuple[str, np.ndarray, float, dict[str, float], bool]] = []
    for label, ctrl in candidates:
        score, metrics = evaluator.evaluate(ctrl)
        out.append((label, ctrl, score, metrics, is_feasible(metrics, criteria)))
    return out


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    tag = args.tag or datetime.now().strftime("run6_%Y%m%d_%H%M%S")
    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = MorphologyBounds(
        x_min=args.x_min, x_max=args.x_max,
        y_min=args.y_min, y_max=args.y_max,
        len_min=args.len_min, len_max=args.len_max,
    )
    criteria = FeasibilityCriteria(
        max_mean_tip_distance=args.max_mean_tip_distance,
        min_contacts=args.min_contacts,
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
    adapt_cfg = Phase1OptimizationConfig(
        iterations=1 if args.fp_adaptation == "sparse-per-morph" else args.adapt_iterations,
        population=5 if args.fp_adaptation == "sparse-per-morph" else args.adapt_population,
        elite_fraction=args.adapt_elite_fraction,
        sigma_init=args.adapt_sigma_init,
        seed=args.adapt_seed,
    )

    global_rows: list[dict[str, object]] = []
    scene_summary: dict[str, dict[str, object]] = {}

    for keyframe_name in args.keyframes:
        spec = KeyframeSpec(
            keyframe=keyframe_name,
            foundational_run_dir=args.foundational_root / keyframe_name,
            max_mean_tip_distance=args.max_mean_tip_distance,
            min_contacts=args.min_contacts,
        )
        foundational_poses = load_foundational_poses(spec.foundational_run_dir, keyframe_name=spec.keyframe)
        base_morph = parse_morphology_from_keyframe(args.scene_xml, keyframe_name=spec.keyframe)

        candidates = sample_morphologies(
            base=base_morph, sample_count=args.samples, rng=rng, bounds=bounds,
            x_perturb=args.x_perturb, y_perturb=args.y_perturb, len_perturb=args.len_perturb,
        )
        if args.morph_sort == "distance":
            candidates.sort(key=lambda m: morph_distance(m, base_morph))

        key_dir = out_dir / spec.keyframe
        gen_dir = key_dir / "generated_mjcf"
        gen_dir.mkdir(parents=True, exist_ok=True)

        all_rows: list[dict[str, object]] = []
        feasible_rows: list[dict[str, object]] = []

        interval_ctrl = np.asarray(foundational_poses[0].finger_ctrl, dtype=np.float64).copy()
        adapt_count = 0
        adapt_seconds = 0.0
        eval_seconds = 0.0

        for idx, morphology in enumerate(candidates):
            scene_xml = gen_dir / f"scene_{spec.keyframe}_{morph_suffix(morphology)}.xml"
            write_rigid_scene_with_object_size(
                base_scene_xml=args.scene_xml,
                output_scene_xml=scene_xml,
                morphology=morphology,
                size_xyz=None,
            )

            evaluator = Phase1GraspEvaluator(scene_xml=scene_xml, keyframe=spec.keyframe, cfg=eval_cfg)

            ctrl_adapted, triggered, secs, source = adapt_foundational_ctrl(
                mode=args.fp_adaptation,
                candidate_idx=idx,
                interval_ctrl=interval_ctrl,
                evaluator=evaluator,
                adapt_cfg=adapt_cfg,
                refresh_interval=args.fp_refresh_interval,
            )
            adapt_count += int(triggered)
            adapt_seconds += secs
            if args.fp_adaptation == "interval-initial-fp" and triggered:
                interval_ctrl = ctrl_adapted.copy()

            t0 = time.perf_counter()
            pool: list[tuple[str, np.ndarray]] = [(f"adapted_{args.fp_adaptation}", ctrl_adapted)]
            pool.extend(
                (p.label, np.asarray(p.finger_ctrl, dtype=np.float64)) for p in foundational_poses
            )
            evals = _evaluate_pool(evaluator, pool, criteria)
            elapsed_eval = float(time.perf_counter() - t0)
            eval_seconds += elapsed_eval

            feasible_hits = [e for e in evals if e[4]]
            any_feasible = len(feasible_hits) > 0
            chosen_label, chosen_ctrl, chosen_score, chosen_metrics, _ = max(
                feasible_hits if feasible_hits else evals, key=lambda e: e[2]
            )

            row: dict[str, object] = {
                "keyframe": spec.keyframe,
                "candidate_id": str(idx),
                "scene_xml": str(scene_xml),
                "fp_adaptation": args.fp_adaptation,
                "selected_foundational_pose": chosen_label,
                "feasible": str(any_feasible),
                "feasible_pose_count": float(len(feasible_hits)),
                "foundational_pose_count": float(len(foundational_poses)),
                "fp_adapt_triggered": str(triggered),
                "fp_adapt_source": source,
                "fp_adapt_seconds": float(secs),
                "evaluation_seconds": float(elapsed_eval),
                "score": float(chosen_score),
                "cube_lift": float(chosen_metrics.get("cube_lift", 0.0)),
                "cube_tip_contacts": float(chosen_metrics.get("cube_tip_contacts", 0.0)),
                "mean_tip_distance": float(chosen_metrics.get("mean_tip_distance", 0.0)),
                "cube_xy_drift": float(chosen_metrics.get("cube_xy_drift", 0.0)),
                "cube_vel_norm": float(chosen_metrics.get("cube_vel_norm", 0.0)),
                "finger_yaw_drift": float(chosen_metrics.get("finger_yaw_drift", 0.0)),
                "finger_flex_drift": float(chosen_metrics.get("finger_flex_drift", 0.0)),
                **morph_row_fields(morphology),
                "morph_distance_from_base": float(morph_distance(morphology, base_morph)),
                "chosen_ctrl_json": json.dumps(chosen_ctrl.tolist()),
            }
            all_rows.append(row)
            global_rows.append(row)
            if any_feasible:
                feasible_rows.append(row)

        pareto_idx = pareto_front_indices(feasible_rows)
        pareto_rows = [feasible_rows[i] for i in pareto_idx]

        write_csv(all_rows, key_dir / "all_candidates.csv")
        write_csv(feasible_rows, key_dir / "feasible_candidates.csv")
        write_csv(pareto_rows, key_dir / "pareto_front.csv")
        plot_feasible_scatter(feasible_rows, pareto_idx, key_dir)

        scene_summary[spec.keyframe] = {
            "total_candidates": len(all_rows),
            "feasible_candidates": len(feasible_rows),
            "pareto_size": len(pareto_rows),
            "fp_adapt_count": adapt_count,
            "fp_adapt_seconds_total": adapt_seconds,
            "evaluation_seconds_total": eval_seconds,
            "mean_eval_seconds": eval_seconds / max(1, len(all_rows)),
        }

        print(
            f"[{spec.keyframe}] total={len(all_rows)} feasible={len(feasible_rows)} "
            f"pareto={len(pareto_rows)} adapt_count={adapt_count}"
        )

    write_csv(global_rows, out_dir / "all_keyframes_candidates.csv")

    summary = {
        "tag": tag,
        "scene_xml": str(args.scene_xml),
        "keyframes": args.keyframes,
        "samples": int(args.samples),
        "seed": int(args.seed),
        "fp_adaptation": args.fp_adaptation,
        "fp_refresh_interval": int(args.fp_refresh_interval),
        "morph_sort": args.morph_sort,
        "bounds": asdict(bounds),
        "perturb": {
            "x": float(args.x_perturb),
            "y": float(args.y_perturb),
            "len": float(args.len_perturb),
        },
        "feasibility": {
            "max_mean_tip_distance": float(args.max_mean_tip_distance),
            "min_contacts": float(args.min_contacts),
        },
        "adapt_cfg": asdict(adapt_cfg),
        "scene_summary": scene_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Run6 sampling complete: {out_dir}")


if __name__ == "__main__":
    main()
