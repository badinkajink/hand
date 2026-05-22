"""Render mp4 videos of top-K morphologies per task and across the eval set.

Inputs:
  <run_dir>/all_candidates_multi.csv     -- aggregate scores per morphology
  <run_dir>/foundational/<task>/<run>/summary.json   -- per-task foundational ctrl
  <run_dir>/generated_mjcf/<task>_<suffix>.xml       -- the rigid scenes already
                                                       written by run18

For each (task, morphology) in the top-K, calls
`Phase1GraspEvaluator.render_rollout(ctrl, ...)`. Uses the task's foundational
ctrl (same as run18's sweep evaluator). Also renders cross-set top-K on each
task using their per-task foundational ctrl.

Outputs:
  <run_dir>/videos/per_task/<task>/top_<rank>_<morph_suffix>.mp4
  <run_dir>/videos/cross_set/top_<rank>_on_<task>/<morph_suffix>.mp4
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from morphohand.optimization.phase1_common import Phase1EvalConfig, Phase1GraspEvaluator  # noqa: E402
from morphohand.optimization.phase1_grasp import Phase1OptimizationConfig, optimize_finger_controls  # noqa: E402
from morphohand.optimization.contact_targets import ContactTargetSet  # noqa: E402
from morphohand.sampling.foundational import load_foundational_poses  # noqa: E402
from morphohand.sampling.morphology import MorphologyValues, morph_suffix  # noqa: E402
from morphohand.sampling.scene import write_rigid_scene_with_object_size  # noqa: E402

# Reuse task specs.
sys.path.insert(0, str(ROOT / "scripts"))
from run18_multi_object_sweep import default_tasks, build_eval_cfg  # noqa: E402


MORPH_KEYS = ["thumb_x", "thumb_y", "thumb_len",
              "index_x", "index_y", "index_len",
              "middle_x", "middle_y", "middle_len"]


def row_to_morphology(row: dict[str, str]) -> MorphologyValues:
    return MorphologyValues(**{k: float(row[k]) for k in MORPH_KEYS})


def render_for(task, morph: MorphologyValues, ctrl: np.ndarray, contact_target_set,
               cfg: Phase1EvalConfig, out_mp4: Path, gen_dir: Path,
               adapt_cfg: Phase1OptimizationConfig | None = None) -> Path:
    scene_out = gen_dir / f"{task.label}_{morph_suffix(morph)}.xml"
    if not scene_out.exists():
        write_rigid_scene_with_object_size(
            base_scene_xml=task.scene_xml,
            output_scene_xml=scene_out,
            morphology=morph,
            object_body_name=task.object_body,
            size_xyz=None,
        )
    evaluator = Phase1GraspEvaluator(
        scene_xml=scene_out,
        keyframe=task.keyframe,
        cfg=cfg,
        contact_target_set=contact_target_set,
        backend="mujoco",
    )
    if adapt_cfg is not None:
        refined = optimize_finger_controls(
            evaluator=evaluator, cfg=adapt_cfg, initial_finger_ctrl=ctrl,
        )
        ctrl = np.asarray(refined["best_finger_ctrl"], dtype=np.float64)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    evaluator.render_rollout(ctrl, out_mp4)
    return out_mp4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--cross-csv", default="all_candidates_multi.csv")
    ap.add_argument("--video-subdir", default="videos")
    ap.add_argument("--adapt-before-render", action="store_true",
                    help="Run a small CEM on each top morphology before rendering")
    ap.add_argument("--adapt-iterations", type=int, default=16)
    ap.add_argument("--adapt-population", type=int, default=36)
    ap.add_argument("--adapt-sigma-init", type=float, default=0.09)
    args = ap.parse_args()

    tasks = default_tasks()
    task_by_label = {t.label: t for t in tasks}

    # Load per-task foundational ctrl
    found_root = args.run_dir / "foundational"
    foundational_ctrl: dict[str, np.ndarray] = {}
    contact_target_sets = {}
    cfgs = {}
    for t in tasks:
        poses = load_foundational_poses(found_root / t.label, keyframe_name=t.keyframe)
        foundational_ctrl[t.label] = np.asarray(poses[0].finger_ctrl, dtype=np.float64).copy()
        cts = (ContactTargetSet.from_yaml(t.contact_targets_yaml)
               if t.contact_targets_yaml and t.contact_targets_yaml.exists() else None)
        contact_target_sets[t.label] = cts
        cfgs[t.label] = build_eval_cfg(t)

    # Read aggregate CSV
    cross_csv = args.run_dir / args.cross_csv
    with cross_csv.open("r") as f:
        cross_rows = list(csv.DictReader(f))

    gen_dir = args.run_dir / "generated_mjcf"
    adapt_cfg = Phase1OptimizationConfig(
        iterations=args.adapt_iterations,
        population=args.adapt_population,
        elite_fraction=0.25,
        sigma_init=args.adapt_sigma_init,
        seed=0,
    ) if args.adapt_before_render else None

    # Per-task top-K
    for t in tasks:
        key = f"score_{t.label}"
        ranked = sorted(cross_rows, key=lambda r: -float(r[key]))[:args.top_k]
        for rank, row in enumerate(ranked):
            morph = row_to_morphology(row)
            out_mp4 = args.run_dir / args.video_subdir / "per_task" / t.label / f"top_{rank:02d}_{morph_suffix(morph)}.mp4"
            if out_mp4.exists():
                print(f"[per_task/{t.label}] rank={rank} exists, skip")
                continue
            print(f"[per_task/{t.label}] rendering rank={rank} score={float(row[key]):+.3f}")
            try:
                render_for(t, morph, foundational_ctrl[t.label],
                           contact_target_sets[t.label], cfgs[t.label], out_mp4, gen_dir,
                           adapt_cfg=adapt_cfg)
            except Exception as e:
                print(f"  FAIL: {e}")

    # Cross-set top-K (by score_mean), render each on EACH task
    ranked = sorted(cross_rows, key=lambda r: -float(r["score_mean"]))[:args.top_k]
    for rank, row in enumerate(ranked):
        morph = row_to_morphology(row)
        for t in tasks:
            out_mp4 = args.run_dir / args.video_subdir / "cross_set" / f"top_{rank:02d}_on_{t.label}" / f"{morph_suffix(morph)}.mp4"
            if out_mp4.exists():
                continue
            print(f"[cross/{rank}] rendering on {t.label} (mean_score={float(row['score_mean']):+.3f})")
            try:
                render_for(t, morph, foundational_ctrl[t.label],
                           contact_target_sets[t.label], cfgs[t.label], out_mp4, gen_dir,
                           adapt_cfg=adapt_cfg)
            except Exception as e:
                print(f"  FAIL: {e}")


if __name__ == "__main__":
    main()
