"""Evaluate Lightning Grasp grasps through Phase1GraspEvaluator.

Reads a JSON of grasps produced by `lightning_grasp_runner.py` and scores
each one in one of two modes:

- `ctrl_only` (default): q is just the actuator target. Cube stays at scene
  keyframe pose. Same protocol as baseline/contact_map/synergy/force_closure
  -- this is the apples-to-apples comparison against existing methods.

- `init_pose`: also initialize `data.qpos` so the hand is *already* at q
  and the cube is at Lightning's expected object_pose (palm-relative). This
  tests the grasp as a snap-to pose, which is what Lightning's pipeline is
  natively producing. Less directly comparable to CEM-based methods but more
  faithful to Lightning's intent.

Usage:
    uv run python scripts/lightning_grasp_eval.py \\
        --grasps-json results/lightning_grasp/cube_grasps.json \\
        --scene-xml assets/mjcf/scene.xml --keyframe open \\
        --output-json results/lightning_grasp/cube_eval.json \\
        --mode ctrl_only
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from morphohand.optimization.phase1_common import Phase1EvalConfig, Phase1GraspEvaluator
from morphohand.sampling.scene import freeze_scene_for_eval


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a MuJoCo (qw, qx, qy, qz) quaternion."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qw, qx, qy, qz], dtype=np.float64)


def _patch_evaluator_for_init_pose(ev: Phase1GraspEvaluator, palm_body_name: str = "palm_pose") -> None:
    """Monkey-patch `ev._reset_to_keyframe` to also apply the per-grasp qpos overrides.

    The overrides are read from `ev._lightning_init` which the caller sets
    before invoking `ev.evaluate(q)`. Use `ev._lightning_init = None` to disable.
    """
    original = ev._reset_to_keyframe
    # MuJoCo freejoint qpos starts at the joint's qposadr; we resolve once.
    cube_jid = mujoco.mj_name2id(ev.model, mujoco.mjtObj.mjOBJ_JOINT, "")  # placeholder; we'll find by free joint instead
    cube_qpos_adr = int(ev.model.body_jntadr[ev.cube_body_id])
    cube_qposadr = int(ev.model.jnt_qposadr[cube_qpos_adr])
    palm_bid = mujoco.mj_name2id(ev.model, mujoco.mjtObj.mjOBJ_BODY, palm_body_name)
    palm_body_pos = ev.model.body_pos[palm_bid].copy() if palm_bid >= 0 else np.zeros(3)

    ev._cube_qposadr = cube_qposadr
    ev._palm_body_pos = palm_body_pos

    def patched():
        original()
        init = getattr(ev, "_lightning_init", None)
        if init is None:
            return
        q_finger = init["q"]
        obj_pose_palm = init["object_pose"]  # 4x4 palm-frame
        # World cube pose = palm_body_pos (palm_pose joints all 0 at reset) @ obj_pose_palm
        world_pos = ev._palm_body_pos + obj_pose_palm[:3, 3]
        world_quat = _rot_to_quat(obj_pose_palm[:3, :3])
        ev.data.qpos[cube_qposadr:cube_qposadr + 3] = world_pos
        ev.data.qpos[cube_qposadr + 3:cube_qposadr + 7] = world_quat
        ev.data.qpos[ev.finger_joint_qpos_ids] = q_finger
        mujoco.mj_forward(ev.model, ev.data)

    ev._reset_to_keyframe = patched


def _frozen_scene(scene_xml: Path, keyframe: str, work_dir: Path) -> Path:
    """Return a frozen-morphology copy of the scene under work_dir/."""
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{scene_xml.stem}__{keyframe}.frozen.xml"
    freeze_scene_for_eval(scene_xml, keyframe, out)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps-json", required=True, type=Path)
    ap.add_argument("--scene-xml", required=True, type=Path)
    ap.add_argument("--keyframe", default="open")
    ap.add_argument("--output-json", required=True, type=Path)
    ap.add_argument("--settle-steps", type=int, default=120)
    ap.add_argument("--lift-steps", type=int, default=80)
    ap.add_argument("--hold-steps", type=int, default=40)
    ap.add_argument("--lift-delta-z", type=float, default=0.05)
    ap.add_argument("--lift-ramp-steps", type=int, default=40)
    ap.add_argument("--max-grasps", type=int, default=0, help="0 = all")
    ap.add_argument("--mode", choices=["ctrl_only", "init_pose"], default="ctrl_only")
    return ap.parse_args()


def main():
    args = parse_args()

    data = json.loads(args.grasps_json.read_text())
    grasps = data["grasps"]
    if args.max_grasps:
        grasps = grasps[: args.max_grasps]

    cfg = Phase1EvalConfig(
        settle_steps=args.settle_steps,
        lift_steps=args.lift_steps,
        hold_steps=args.hold_steps,
        lift_delta_z=args.lift_delta_z,
        lift_ramp_steps=args.lift_ramp_steps,
    )

    frozen = _frozen_scene(args.scene_xml, args.keyframe, args.output_json.parent / "frozen_scenes")
    evaluator = Phase1GraspEvaluator(
        scene_xml=frozen, keyframe=args.keyframe, cfg=cfg, backend="mujoco"
    )

    expected_joints = evaluator.finger_joint_names
    incoming_joints = grasps[0]["active_joints"]
    if list(incoming_joints) != list(expected_joints):
        raise SystemExit(
            f"Joint-order mismatch:\n  grasps: {incoming_joints}\n  scene:  {expected_joints}"
        )

    if args.mode == "init_pose":
        _patch_evaluator_for_init_pose(evaluator)

    results = []
    t0 = time.time()
    for i, g in enumerate(grasps):
        q = np.asarray(g["q"], dtype=np.float64)
        if args.mode == "init_pose":
            evaluator._lightning_init = {
                "q": q,
                "object_pose": np.asarray(g["object_pose"], dtype=np.float64),
            }
        score, diag = evaluator.evaluate(q)
        results.append({"idx": i, "score": float(score), "q": g["q"], "diagnostics": diag})
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(grasps)}] score={score:+.3f}  ({elapsed:.1f}s)")
    t_eval = time.time() - t0

    scores = np.array([r["score"] for r in results])
    summary = {
        "scene_xml": str(args.scene_xml),
        "keyframe": args.keyframe,
        "frozen_scene": str(frozen),
        "mode": args.mode,
        "n_grasps_generated": data["n_grasps"],
        "n_grasps_evaluated": len(results),
        "lightning_wall_time_s": data["wall_time_s"],
        "eval_wall_time_s": t_eval,
        "best_score": float(scores.max()),
        "median_score": float(np.median(scores)),
        "mean_score": float(scores.mean()),
        "std_score": float(scores.std()),
        "best_idx": int(scores.argmax()),
        "best_q": results[int(scores.argmax())]["q"],
        "per_grasp": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(
        f"\nDone. best={summary['best_score']:+.3f}  median={summary['median_score']:+.3f}  "
        f"mean={summary['mean_score']:+.3f}  ({len(results)} grasps, {t_eval:.1f}s)\n"
        f"Wrote {args.output_json}"
    )


if __name__ == "__main__":
    main()
