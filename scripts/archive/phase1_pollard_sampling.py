from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
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
)
from morphohand.sampling import (  # noqa: E402
    FeasibilityCriteria,
    MorphologyBounds,
    is_feasible,
    load_base_morphology,
    load_foundational_poses_with_scene,
    morph_row_fields,
    pareto_front_indices,
    plot_feasible_scatter,
    sample_morphologies,
    write_csv,
)
from morphohand.tools.morphology_xml import (  # noqa: E402
    create_rigid_hand_and_scene_xmls,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pollard-style morphology sampling on the block scene: sample candidate "
            "morphologies, filter by foundational-pose feasibility, export Pareto fronts."
        )
    )
    parser.add_argument(
        "--foundational-run-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase1" / "run_20260410_163959",
    )
    parser.add_argument("--base-hand-xml", type=Path, default=PROJECT_ROOT / "assets" / "mjcf" / "hand.xml")
    parser.add_argument("--base-scene-xml", type=Path, default=PROJECT_ROOT / "assets" / "mjcf" / "scene.xml")
    parser.add_argument("--keyframe", default="open")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x-perturb", type=float, default=0.012)
    parser.add_argument("--y-perturb", type=float, default=0.012)
    parser.add_argument("--len-perturb", type=float, default=0.012)
    parser.add_argument("--x-min", type=float, default=-0.03)
    parser.add_argument("--x-max", type=float, default=0.03)
    parser.add_argument("--y-min", type=float, default=-0.03)
    parser.add_argument("--y-max", type=float, default=0.03)
    parser.add_argument("--len-min", type=float, default=0.0)
    parser.add_argument("--len-max", type=float, default=0.035)
    parser.add_argument("--max-mean-tip-distance", type=float, default=0.012)
    parser.add_argument("--min-contacts", type=float, default=2.0)
    parser.add_argument(
        "--feasible-rule",
        choices=["any", "all"],
        default="any",
        help="Whether a morphology must satisfy any or all foundational poses.",
    )
    parser.add_argument("--settle-steps", type=int, default=240)
    parser.add_argument("--lift-steps", type=int, default=220)
    parser.add_argument("--hold-steps", type=int, default=140)
    parser.add_argument("--lift-delta-z", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "phase1")
    parser.add_argument("--tag", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    tag = args.tag or datetime.now().strftime("run_%Y%m%d_%H%M%S_pollard_block")
    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_xml_dir = out_dir / "generated_mjcf"

    foundational_poses, foundational_scene = load_foundational_poses_with_scene(
        args.foundational_run_dir
    )
    base_morphology = load_base_morphology(foundational_scene, keyframe_name=args.keyframe)

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

    all_rows: list[dict[str, object]] = []
    feasible_rows: list[dict[str, object]] = []

    for idx, morphology in enumerate(candidates):
        _, scene_xml = create_rigid_hand_and_scene_xmls(
            base_hand_xml_path=args.base_hand_xml,
            base_scene_xml_path=args.base_scene_xml,
            morphology=morphology,
            output_dir=generated_xml_dir,
            hand_prefix="hand",
            scene_prefix="scene",
        )

        evaluator = Phase1GraspEvaluator(scene_xml=scene_xml, keyframe=args.keyframe, cfg=eval_cfg)
        per_pose: list[tuple[object, float, dict[str, float], bool]] = []
        for pose in foundational_poses:
            score, metrics = evaluator.evaluate(pose.finger_ctrl)
            per_pose.append((pose, score, metrics, is_feasible(metrics, criteria)))

        feasible_hits = [t for t in per_pose if t[3]]
        satisfied = (
            len(feasible_hits) >= 1
            if args.feasible_rule == "any"
            else len(feasible_hits) == len(foundational_poses)
        )

        chosen_pool = feasible_hits if feasible_hits else per_pose
        chosen_pose, chosen_score, chosen_metrics, _ = max(chosen_pool, key=lambda t: t[1])

        row: dict[str, object] = {
            "candidate_id": str(idx),
            "scene_xml": str(scene_xml),
            "selected_foundational_pose": chosen_pose.label,
            "feasible_rule": args.feasible_rule,
            "feasible": str(bool(satisfied)),
            "feasible_pose_count": float(len(feasible_hits)),
            "foundational_pose_count": float(len(foundational_poses)),
            "score": float(chosen_score),
            "cube_lift": float(chosen_metrics.get("cube_lift", 0.0)),
            "cube_tip_contacts": float(chosen_metrics.get("cube_tip_contacts", 0.0)),
            "mean_tip_distance": float(chosen_metrics.get("mean_tip_distance", 0.0)),
            "cube_vel_norm": float(chosen_metrics.get("cube_vel_norm", 0.0)),
            **morph_row_fields(morphology),
        }
        all_rows.append(row)
        if satisfied:
            feasible_rows.append(row)

    pareto_idx = pareto_front_indices(feasible_rows)
    pareto_rows = [feasible_rows[i] for i in pareto_idx]

    write_csv(all_rows, out_dir / "all_candidates.csv")
    write_csv(feasible_rows, out_dir / "feasible_candidates.csv")
    write_csv(pareto_rows, out_dir / "pareto_front.csv")
    plot_feasible_scatter(feasible_rows, pareto_idx, out_dir)

    summary = {
        "tag": tag,
        "foundational_run_dir": str(args.foundational_run_dir),
        "foundational_scene_xml": str(foundational_scene),
        "foundational_pose_labels": [p.label for p in foundational_poses],
        "foundational_pose_scores": {p.label: p.score for p in foundational_poses},
        "base_morphology": asdict(base_morphology),
        "bounds": asdict(bounds),
        "sampling": {
            "samples": int(args.samples),
            "seed": int(args.seed),
            "x_perturb": float(args.x_perturb),
            "y_perturb": float(args.y_perturb),
            "len_perturb": float(args.len_perturb),
        },
        "feasibility": {
            "max_mean_tip_distance": float(args.max_mean_tip_distance),
            "min_contacts": float(args.min_contacts),
            "rule": args.feasible_rule,
        },
        "counts": {
            "total_candidates": len(all_rows),
            "feasible_candidates": len(feasible_rows),
            "pareto_size": len(pareto_rows),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Pollard sampling run complete: {out_dir}")
    print(f"Total candidates: {len(all_rows)}")
    print(f"Feasible candidates: {len(feasible_rows)}")
    print(f"Pareto front size: {len(pareto_rows)}")


if __name__ == "__main__":
    main()
