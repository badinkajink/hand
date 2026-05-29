from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

import sys
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from morphohand.optimization.phase1_common import Phase1EvalConfig, Phase1GraspEvaluator
from morphohand.sampling.scene import freeze_scene_for_eval


def _load_summary(run_dir: Path) -> dict:
    with (run_dir / "summary.json").open() as f:
        return json.load(f)


def _resolve_frozen_scene(run_dir: Path, summary: dict) -> Path:
    frozen = run_dir / "frozen_scene.xml"
    if frozen.exists():
        return frozen
    scene_xml = Path(summary["scene_xml"]).resolve()
    keyframe = summary.get("keyframe", "open")
    frozen = run_dir / f"frozen_{scene_xml.stem}.xml"
    freeze_scene_for_eval(scene_xml, keyframe, frozen)
    return frozen


def _eval_cfg_from_summary(summary: dict) -> Phase1EvalConfig:
    cfg = Phase1EvalConfig()
    for key, value in (summary.get("eval_config", {}) or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference-only playback via Phase1 evaluator.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["mujoco", "mjwarp", "comfree-warp"], default="mujoco")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--backend-sync-interval", type=int, default=1)
    parser.add_argument("--metric-sample-interval", type=int, default=1)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    summary = _load_summary(run_dir)
    frozen = _resolve_frozen_scene(run_dir, summary)
    keyframe = summary.get("keyframe", "open")

    npz = np.load(run_dir / "best_rollout.npz")
    finger_ctrl = np.asarray(npz["best_finger_ctrl"], dtype=np.float64).reshape(-1)

    cfg = _eval_cfg_from_summary(summary)
    evaluator = Phase1GraspEvaluator(
        scene_xml=frozen,
        keyframe=keyframe,
        cfg=cfg,
        backend=args.backend,
        backend_sync_interval=args.backend_sync_interval,
        metric_sample_interval=args.metric_sample_interval,
    )

    score, metrics = evaluator.evaluate(finger_ctrl)
    print(f"score={score:.4f}")
    print("metrics:")
    for k in sorted(metrics.keys()):
        print(f"  {k}: {metrics[k]}")

    rollout = evaluator.rollout(finger_ctrl)
    contacts = rollout["contacts"]
    cube_z = rollout["cube_z"]
    frac_2plus = float(np.mean(contacts >= 2))
    frac_3 = float(np.mean(contacts >= 3))
    print("contact_summary:")
    print(f"  mean_contacts: {float(np.mean(contacts)):.3f}")
    print(f"  frac_steps_2plus: {frac_2plus:.3f}")
    print(f"  frac_steps_3: {frac_3:.3f}")
    print(f"  cube_z_peak: {float(np.max(cube_z)):.4f}")

    if args.skip_render:
        return

    output = args.output
    if output is None:
        output = run_dir / f"reference_playback_{args.backend}.mp4"
    evaluator.render_rollout(
        finger_ctrl,
        output_path=output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        frame_stride=args.frame_stride,
    )
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
