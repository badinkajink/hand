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
    parser.add_argument(
        "--keyframes",
        nargs="+",
        default=["open_flat", "open_vertical", "open_90vertical"],
    )
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
        "--objective-weight-finger-persistence-imbalance-penalty",
        type=float,
        default=1.2,
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


def _write_scene_with_morphology(
    base_scene_xml: Path,
    output_scene_xml: Path,
    morphology: MorphologyValues,
) -> None:
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

    poses = sorted(poses, key=lambda p: p.score, reverse=True)
    return poses


def _is_feasible(metrics: dict[str, float], max_mean_tip_distance: float, min_contacts: float) -> bool:
    return (
        float(metrics.get("mean_tip_distance", np.inf)) <= max_mean_tip_distance
        and float(metrics.get("cube_tip_contacts", 0.0)) >= min_contacts
    )


def _pareto_front_indices(rows: list[dict[str, float | str]]) -> list[int]:
    if not rows:
        return []

    max_keys = ("score", "cube_lift", "cube_tip_contacts")
    min_keys = ("mean_tip_distance", "cube_vel_norm")
    dominated = [False] * len(rows)

    for i, a in enumerate(rows):
        if dominated[i]:
            continue
        for j, b in enumerate(rows):
            if i == j or dominated[i]:
                continue
            b_not_worse = all(float(b[k]) >= float(a[k]) for k in max_keys) and all(
                float(b[k]) <= float(a[k]) for k in min_keys
            )
            b_strict = any(float(b[k]) > float(a[k]) for k in max_keys) or any(
                float(b[k]) < float(a[k]) for k in min_keys
            )
            if b_not_worse and b_strict:
                dominated[i] = True

    return [i for i, d in enumerate(dominated) if not d]


def _write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_scene_results(feasible_rows: list[dict[str, float | str]], pareto_idx: list[int], out_dir: Path) -> None:
    if not feasible_rows:
        return

    dist = np.asarray([float(r["mean_tip_distance"]) for r in feasible_rows], dtype=np.float64)
    lift = np.asarray([float(r["cube_lift"]) for r in feasible_rows], dtype=np.float64)
    score = np.asarray([float(r["score"]) for r in feasible_rows], dtype=np.float64)

    plt.figure(figsize=(8, 4.8))
    sc = plt.scatter(dist, lift, c=score, s=24, alpha=0.85, cmap="viridis")
    if pareto_idx:
        plt.scatter(
            dist[pareto_idx],
            lift[pareto_idx],
            facecolors="none",
            edgecolors="red",
            s=80,
            linewidths=1.2,
            label="Pareto",
        )
        plt.legend()
    plt.xlabel("Mean tip distance")
    plt.ylabel("Lift")
    plt.title("Feasible morphology scatter")
    plt.grid(True, alpha=0.25)
    plt.colorbar(sc, label="Score")
    plt.tight_layout()
    plt.savefig(out_dir / "feasible_scatter_lift_vs_distance.png", dpi=180)
    plt.close()


def _adapt_ctrl(
    adaptation_mode: str,
    candidate_idx: int,
    interval_ctrl: np.ndarray,
    evaluator: Phase1GraspEvaluator,
    adapt_cfg: Phase1OptimizationConfig,
    refresh_interval: int,
) -> tuple[np.ndarray, bool, float, str]:
    if adaptation_mode == "none":
        return interval_ctrl, False, 0.0, "none"

    should_refresh = False
    source = "reuse"
    if adaptation_mode == "interval-initial-fp":
        should_refresh = (candidate_idx % max(1, refresh_interval)) == 0
        source = "interval_fp"
    elif adaptation_mode == "sparse-per-morph":
        should_refresh = True
        source = "sparse"

    if not should_refresh:
        return interval_ctrl, False, 0.0, source

    t0 = time.perf_counter()
    refined = optimize_finger_controls(evaluator=evaluator, cfg=adapt_cfg, initial_finger_ctrl=interval_ctrl)
    elapsed = float(time.perf_counter() - t0)
    ctrl = np.asarray(refined["best_finger_ctrl"], dtype=np.float64)
    return ctrl, True, elapsed, source


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    tag = args.tag or datetime.now().strftime("run6_%Y%m%d_%H%M%S")
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
    adapt_cfg = Phase1OptimizationConfig(
        iterations=1 if args.fp_adaptation == "sparse-per-morph" else args.adapt_iterations,
        population=5 if args.fp_adaptation == "sparse-per-morph" else args.adapt_population,
        elite_fraction=args.adapt_elite_fraction,
        sigma_init=args.adapt_sigma_init,
        seed=args.adapt_seed,
    )

    global_rows: list[dict[str, float | str]] = []
    scene_summary: dict[str, dict[str, float | int | str]] = {}

    for keyframe_name in args.keyframes:
        spec = KeyframeSpec(
            keyframe=keyframe_name,
            foundational_run_dir=args.foundational_root / keyframe_name,
            max_mean_tip_distance=args.max_mean_tip_distance,
            min_contacts=args.min_contacts,
        )

        foundational_poses = _load_foundational_poses(spec.foundational_run_dir, keyframe_name=spec.keyframe)
        base_morph = _extract_keyframe_morphology(args.scene_xml, keyframe_name=spec.keyframe)
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

        key_dir = out_dir / spec.keyframe
        gen_dir = key_dir / "generated_mjcf"
        gen_dir.mkdir(parents=True, exist_ok=True)

        all_rows: list[dict[str, float | str]] = []
        feasible_rows: list[dict[str, float | str]] = []

        baseline_ctrl = np.asarray(foundational_poses[0].finger_ctrl, dtype=np.float64)
        interval_ctrl = baseline_ctrl.copy()
        adapt_count = 0
        adapt_seconds = 0.0
        eval_seconds = 0.0

        for idx, morphology in enumerate(candidates):
            suffix = (
                f"t{morphology.thumb_x:+0.4f}_{morphology.thumb_y:+0.4f}_{morphology.thumb_len:+0.4f}_"
                f"i{morphology.index_x:+0.4f}_{morphology.index_y:+0.4f}_{morphology.index_len:+0.4f}_"
                f"m{morphology.middle_x:+0.4f}_{morphology.middle_y:+0.4f}_{morphology.middle_len:+0.4f}"
            ).replace("+", "p").replace("-", "n").replace(".", "d")
            scene_xml = gen_dir / f"scene_{spec.keyframe}_{suffix}.xml"
            _write_scene_with_morphology(args.scene_xml, scene_xml, morphology)

            evaluator = Phase1GraspEvaluator(scene_xml=scene_xml, keyframe=spec.keyframe, cfg=eval_cfg)

            ctrl_adapted, triggered, secs, source = _adapt_ctrl(
                adaptation_mode=args.fp_adaptation,
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
            evals: list[tuple[str, np.ndarray, float, dict[str, float], bool]] = []
            adapted_score, adapted_metrics = evaluator.evaluate(ctrl_adapted)
            adapted_feasible = _is_feasible(
                adapted_metrics,
                max_mean_tip_distance=spec.max_mean_tip_distance,
                min_contacts=spec.min_contacts,
            )
            evals.append((f"adapted_{args.fp_adaptation}", ctrl_adapted, adapted_score, adapted_metrics, adapted_feasible))

            for pose in foundational_poses:
                s, m = evaluator.evaluate(pose.finger_ctrl)
                feasible = _is_feasible(
                    m,
                    max_mean_tip_distance=spec.max_mean_tip_distance,
                    min_contacts=spec.min_contacts,
                )
                evals.append((pose.label, np.asarray(pose.finger_ctrl, dtype=np.float64), s, m, feasible))

            elapsed_eval = float(time.perf_counter() - t0)
            eval_seconds += elapsed_eval

            feasible_hits = [e for e in evals if e[4]]
            is_feasible = len(feasible_hits) > 0
            chosen_pool = feasible_hits if feasible_hits else evals
            chosen_label, chosen_ctrl, chosen_score, chosen_metrics, _ = max(chosen_pool, key=lambda e: e[2])

            row = {
                "keyframe": spec.keyframe,
                "candidate_id": str(idx),
                "scene_xml": str(scene_xml),
                "fp_adaptation": args.fp_adaptation,
                "selected_foundational_pose": chosen_label,
                "feasible": str(is_feasible),
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
                "chosen_ctrl_json": json.dumps(chosen_ctrl.tolist()),
            }
            all_rows.append(row)
            global_rows.append(row)
            if is_feasible:
                feasible_rows.append(row)

        pareto_idx = _pareto_front_indices(feasible_rows)
        pareto_rows = [feasible_rows[i] for i in pareto_idx]

        _write_csv(all_rows, key_dir / "all_candidates.csv")
        _write_csv(feasible_rows, key_dir / "feasible_candidates.csv")
        _write_csv(pareto_rows, key_dir / "pareto_front.csv")
        _plot_scene_results(feasible_rows, pareto_idx, key_dir)

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

    _write_csv(global_rows, out_dir / "all_keyframes_candidates.csv")

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
        "adapt_cfg": {
            "iterations": int(adapt_cfg.iterations),
            "population": int(adapt_cfg.population),
            "elite_fraction": float(adapt_cfg.elite_fraction),
            "sigma_init": float(adapt_cfg.sigma_init),
            "seed": int(adapt_cfg.seed),
        },
        "scene_summary": scene_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Run6 sampling complete: {out_dir}")


if __name__ == "__main__":
    main()
