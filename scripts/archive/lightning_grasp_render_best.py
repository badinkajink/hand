"""Render MP4s of the best Lightning grasp under both eval modes.

Reads a Lightning eval JSON (produced by `lightning_grasp_eval.py`) and
re-runs the best grasp through `Phase1GraspEvaluator.render_rollout` to
write an MP4 of the settle/lift/hold rollout.
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from morphohand.optimization.phase1_common import Phase1EvalConfig, Phase1GraspEvaluator
from morphohand.sampling.scene import freeze_scene_for_eval

# Reuse the qpos-override patch from the eval script.
from lightning_grasp_eval import _patch_evaluator_for_init_pose  # type: ignore


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps-json", required=True, type=Path,
                    help="raw Lightning runner output (need object_pose for init_pose mode)")
    ap.add_argument("--eval-json", required=True, type=Path,
                    help="lightning_grasp_eval.py output with best_idx and best_q")
    ap.add_argument("--scene-xml", required=True, type=Path)
    ap.add_argument("--keyframe", default="open")
    ap.add_argument("--output-mp4", required=True, type=Path)
    ap.add_argument("--mode", choices=["ctrl_only", "init_pose"], required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    eval_data = json.loads(args.eval_json.read_text())
    grasps_data = json.loads(args.grasps_json.read_text())
    best_idx = int(eval_data["best_idx"])
    best_q = np.asarray(eval_data["best_q"], dtype=np.float64)
    best_score = float(eval_data["best_score"])

    cfg = Phase1EvalConfig(settle_steps=120, lift_steps=80, hold_steps=40,
                           lift_delta_z=0.05, lift_ramp_steps=40)
    work_dir = args.output_mp4.parent / "frozen_scenes"
    work_dir.mkdir(parents=True, exist_ok=True)
    frozen = work_dir / f"{args.scene_xml.stem}__{args.keyframe}.frozen.xml"
    freeze_scene_for_eval(args.scene_xml, args.keyframe, frozen)

    ev = Phase1GraspEvaluator(scene_xml=frozen, keyframe=args.keyframe, cfg=cfg, backend="mujoco")
    if args.mode == "init_pose":
        _patch_evaluator_for_init_pose(ev)
        obj_pose = np.asarray(grasps_data["grasps"][best_idx]["object_pose"], dtype=np.float64)
        ev._lightning_init = {"q": best_q, "object_pose": obj_pose}

    args.output_mp4.parent.mkdir(parents=True, exist_ok=True)
    out = ev.render_rollout(best_q, args.output_mp4, frame_stride=2)
    print(f"Wrote {out}  (best score {best_score:+.3f}, idx {best_idx}, mode {args.mode})")


if __name__ == "__main__":
    main()
