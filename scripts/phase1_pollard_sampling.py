from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import sys
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
)
from morphohand.tools.morphology_xml import (  # noqa: E402
    MorphologyValues,
    create_rigid_hand_and_scene_xmls,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pollard-style morphology sampling on the block scene: sample candidate morphologies, "
            "filter by foundational-pose feasibility, and export Pareto fronts."
        )
    )
    parser.add_argument(
        "--foundational-run-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase1" / "run_20260410_163959",
        help="Run directory that contains prior CEM subruns with summary.json files.",
    )
    parser.add_argument(
        "--base-hand-xml",
        type=Path,
        default=PROJECT_ROOT / "assets" / "mjcf" / "hand.xml",
        help="Template hand XML with morphology joints.",
    )
    parser.add_argument(
        "--base-scene-xml",
        type=Path,
        default=PROJECT_ROOT / "assets" / "mjcf" / "scene.xml",
        help="Template scene XML with morphology joints.",
    )
    parser.add_argument("--keyframe", default="open")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--x-perturb",
        type=float,
        default=0.012,
        help="Uniform perturbation half-width for each finger x value.",
    )
    parser.add_argument(
        "--y-perturb",
        type=float,
        default=0.012,
        help="Uniform perturbation half-width for each finger y value.",
    )
    parser.add_argument(
        "--len-perturb",
        type=float,
        default=0.012,
        help="Uniform perturbation half-width for each finger length value.",
    )
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase1",
        help="Base output directory.",
    )
    parser.add_argument("--tag", default=None)
    return parser


def _parse_open_keyframe_morphology(scene_xml: Path) -> MorphologyValues:
    root = ET.parse(scene_xml).getroot()
    keyframe = root.find("keyframe")
    if keyframe is None:
        raise ValueError(f"No <keyframe> block in scene XML: {scene_xml}")

    open_key = None
    for key in keyframe.findall("key"):
        if key.get("name") == "open":
            open_key = key
            break

    if open_key is None:
        raise ValueError(f"No keyframe named 'open' in scene XML: {scene_xml}")

    qpos_raw = open_key.get("qpos")
    if not qpos_raw:
        raise ValueError(f"Open keyframe in {scene_xml} has no qpos")

    qpos = [float(v) for v in qpos_raw.replace("\n", " ").split()]
    return extract_morphology_from_qpos(qpos=qpos, has_scene_prefix=True)


def _decode_token(token: str) -> float:
    if token.startswith("p"):
        sign = 1.0
        body = token[1:]
    elif token.startswith("n"):
        sign = -1.0
        body = token[1:]
    else:
        raise ValueError(f"Invalid morphology token sign: {token}")

    if "d" not in body:
        raise ValueError(f"Invalid morphology token format: {token}")
    return sign * float(body.replace("d", ".", 1))


def _parse_morphology_from_generated_filename(scene_xml: Path) -> MorphologyValues:
    stem = scene_xml.stem
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Scene filename does not contain morphology suffix: {scene_xml.name}")

    t_part = next((p for p in parts if p.startswith("t")), None)
    i_part = next((p for p in parts if p.startswith("i")), None)
    m_part = next((p for p in parts if p.startswith("m")), None)
    if t_part is None or i_part is None or m_part is None:
        raise ValueError(f"Scene filename missing t/i/m morphology groups: {scene_xml.name}")

    def split_triplet(group: str) -> tuple[float, float, float]:
        payload = group[1:]
        if len(payload) % 7 != 0 or len(payload) < 21:
            raise ValueError(f"Unexpected morphology token length in '{group}'")
        chunks = [payload[k : k + 7] for k in range(0, 21, 7)]
        return (_decode_token(chunks[0]), _decode_token(chunks[1]), _decode_token(chunks[2]))

    tx, ty, tl = split_triplet(t_part)
    ix, iy, il = split_triplet(i_part)
    mx, my, ml = split_triplet(m_part)
    return MorphologyValues(
        thumb_x=tx,
        thumb_y=ty,
        thumb_len=tl,
        index_x=ix,
        index_y=iy,
        index_len=il,
        middle_x=mx,
        middle_y=my,
        middle_len=ml,
    )


def _load_foundational_poses(foundational_run_dir: Path) -> tuple[list[FoundationalPose], Path]:
    summaries: list[tuple[dict, Path]] = []
    for subdir in sorted(foundational_run_dir.iterdir()):
        if not subdir.is_dir():
            continue
        summary_path = subdir / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append((payload, subdir))

    if not summaries:
        raise ValueError(f"No subrun summary files found in {foundational_run_dir}")

    poses: list[FoundationalPose] = []
    scene_xml_path: Path | None = None
    for payload, subdir in summaries:
        ctrl = np.asarray(payload["best_finger_ctrl"], dtype=np.float64)
        label = subdir.name
        score = float(payload.get("best_score", float("-inf")))
        poses.append(FoundationalPose(label=label, finger_ctrl=ctrl, score=score))

        current_scene = Path(payload["scene_xml"])
        if scene_xml_path is None:
            scene_xml_path = current_scene
        elif current_scene != scene_xml_path:
            raise ValueError(
                "Foundational run has mixed scene_xml values. Use runs generated from a single block scene."
            )

    poses = sorted(poses, key=lambda p: p.score, reverse=True)
    assert scene_xml_path is not None
    return poses, scene_xml_path


def _load_base_morphology(scene_xml: Path) -> MorphologyValues:
    try:
        return _parse_open_keyframe_morphology(scene_xml)
    except Exception:
        return _parse_morphology_from_generated_filename(scene_xml)


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

    candidates: list[MorphologyValues] = [base]
    seen: set[tuple[float, ...]] = {
        (
            round(base.thumb_x, 6),
            round(base.thumb_y, 6),
            round(base.thumb_len, 6),
            round(base.index_x, 6),
            round(base.index_y, 6),
            round(base.index_len, 6),
            round(base.middle_x, 6),
            round(base.middle_y, 6),
            round(base.middle_len, 6),
        )
    }

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
        key = (
            round(proposal.thumb_x, 6),
            round(proposal.thumb_y, 6),
            round(proposal.thumb_len, 6),
            round(proposal.index_x, 6),
            round(proposal.index_y, 6),
            round(proposal.index_len, 6),
            round(proposal.middle_x, 6),
            round(proposal.middle_y, 6),
            round(proposal.middle_len, 6),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(proposal)

    return candidates


def _is_feasible(metrics: dict[str, float], max_mean_tip_distance: float, min_contacts: float) -> bool:
    return (
        float(metrics.get("mean_tip_distance", np.inf)) <= max_mean_tip_distance
        and float(metrics.get("cube_tip_contacts", 0.0)) >= min_contacts
    )


def _pareto_front_indices(rows: list[dict[str, float]]) -> list[int]:
    if not rows:
        return []

    max_keys = ("score", "cube_lift", "cube_tip_contacts")
    min_keys = ("mean_tip_distance", "cube_vel_norm")

    n = len(rows)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue
        a = rows[i]
        for j in range(n):
            if i == j or dominated[i]:
                continue
            b = rows[j]

            b_not_worse = all(b[k] >= a[k] for k in max_keys) and all(b[k] <= a[k] for k in min_keys)
            b_strict_better = any(b[k] > a[k] for k in max_keys) or any(b[k] < a[k] for k in min_keys)
            if b_not_worse and b_strict_better:
                dominated[i] = True

    return [i for i in range(n) if not dominated[i]]


def _write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_results(
    feasible_rows: list[dict[str, float | str]], pareto_indices: list[int], out_dir: Path
) -> None:
    if not feasible_rows:
        return

    lift = np.asarray([float(r["cube_lift"]) for r in feasible_rows], dtype=np.float64)
    dist = np.asarray([float(r["mean_tip_distance"]) for r in feasible_rows], dtype=np.float64)
    contacts = np.asarray([float(r["cube_tip_contacts"]) for r in feasible_rows], dtype=np.float64)
    score = np.asarray([float(r["score"]) for r in feasible_rows], dtype=np.float64)

    plt.figure(figsize=(8, 4.8))
    scatter = plt.scatter(dist, lift, c=score, s=28, alpha=0.8, cmap="viridis")
    if pareto_indices:
        plt.scatter(
            dist[pareto_indices],
            lift[pareto_indices],
            facecolors="none",
            edgecolors="red",
            s=80,
            linewidths=1.2,
            label="Pareto front",
        )
        plt.legend()
    plt.xlabel("Mean tip distance (lower is better)")
    plt.ylabel("Cube lift (higher is better)")
    plt.title("Feasible Morphologies: Foundational-Pose Sampling")
    plt.grid(True, alpha=0.25)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Score")
    plt.tight_layout()
    plt.savefig(out_dir / "feasible_scatter_lift_vs_distance.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.scatter(contacts, lift, c=score, s=28, alpha=0.8, cmap="plasma")
    if pareto_indices:
        plt.scatter(
            contacts[pareto_indices],
            lift[pareto_indices],
            facecolors="none",
            edgecolors="black",
            s=80,
            linewidths=1.2,
        )
    plt.xlabel("Cube-tip contacts")
    plt.ylabel("Cube lift")
    plt.title("Feasible Morphologies: Lift vs Contacts")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "feasible_scatter_lift_vs_contacts.png", dpi=180)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    tag = args.tag or datetime.now().strftime("run_%Y%m%d_%H%M%S_pollard_block")
    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_xml_dir = out_dir / "generated_mjcf"

    foundational_poses, foundational_scene = _load_foundational_poses(args.foundational_run_dir)
    base_morphology = _load_base_morphology(foundational_scene)

    bounds = MorphologyBounds(
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        len_min=args.len_min,
        len_max=args.len_max,
    )

    candidates = _sample_morphologies(
        base=base_morphology,
        sample_count=args.samples,
        rng=rng,
        bounds=bounds,
        x_perturb=args.x_perturb,
        y_perturb=args.y_perturb,
        len_perturb=args.len_perturb,
    )

    eval_cfg = Phase1EvalConfig(
        settle_steps=args.settle_steps,
        lift_steps=args.lift_steps,
        hold_steps=args.hold_steps,
        lift_delta_z=args.lift_delta_z,
    )

    all_rows: list[dict[str, float | str]] = []
    feasible_rows: list[dict[str, float | str]] = []

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
        per_pose_metrics: list[tuple[FoundationalPose, float, dict[str, float], bool]] = []

        for pose in foundational_poses:
            score, metrics = evaluator.evaluate(pose.finger_ctrl)
            feasible = _is_feasible(
                metrics,
                max_mean_tip_distance=args.max_mean_tip_distance,
                min_contacts=args.min_contacts,
            )
            per_pose_metrics.append((pose, score, metrics, feasible))

        feasible_hits = [t for t in per_pose_metrics if t[3]]
        is_feasible = len(feasible_hits) >= 1 if args.feasible_rule == "any" else len(feasible_hits) == len(foundational_poses)

        chosen_pool = feasible_hits if feasible_hits else per_pose_metrics
        chosen_pose, chosen_score, chosen_metrics, _ = max(chosen_pool, key=lambda t: t[1])

        row = {
            "candidate_id": str(idx),
            "scene_xml": str(scene_xml),
            "selected_foundational_pose": chosen_pose.label,
            "feasible_rule": args.feasible_rule,
            "feasible": str(bool(is_feasible)),
            "feasible_pose_count": float(len(feasible_hits)),
            "foundational_pose_count": float(len(foundational_poses)),
            "score": float(chosen_score),
            "cube_lift": float(chosen_metrics.get("cube_lift", 0.0)),
            "cube_tip_contacts": float(chosen_metrics.get("cube_tip_contacts", 0.0)),
            "mean_tip_distance": float(chosen_metrics.get("mean_tip_distance", 0.0)),
            "cube_vel_norm": float(chosen_metrics.get("cube_vel_norm", 0.0)),
            "thumb_x": float(morphology.thumb_x),
            "thumb_y": float(morphology.thumb_y),
            "thumb_len": float(morphology.thumb_len),
            "index_x": float(morphology.index_x),
            "index_y": float(morphology.index_y),
            "index_len": float(morphology.index_len),
            "middle_x": float(morphology.middle_x),
            "middle_y": float(morphology.middle_y),
            "middle_len": float(morphology.middle_len),
        }
        all_rows.append(row)
        if is_feasible:
            feasible_rows.append(row)

    pareto_rows: list[dict[str, float | str]] = []
    pareto_indices = _pareto_front_indices(feasible_rows)
    for i in pareto_indices:
        pareto_rows.append(feasible_rows[i])

    _write_csv(all_rows, out_dir / "all_candidates.csv")
    _write_csv(feasible_rows, out_dir / "feasible_candidates.csv")
    _write_csv(pareto_rows, out_dir / "pareto_front.csv")
    _plot_results(feasible_rows, pareto_indices, out_dir)

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
