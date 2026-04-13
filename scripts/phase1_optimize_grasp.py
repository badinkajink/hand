from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from morphohand.optimization.phase1_grasp import (  # noqa: E402
    Phase1AutodiffConfig,
    Phase1EvalConfig,
    Phase1GraspEvaluator,
    Phase1OptimizationConfig,
    optimize_finger_controls_autodiff,
    optimize_finger_controls,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 1 inner-loop grasp optimization.")
    parser.add_argument(
        "--scene-xml",
        type=Path,
        default=PROJECT_ROOT
        / "assets"
        / "mjcf"
        / "generated"
        / "scene_tp0d0000p0d0200p0d0000_ip0d0100n0d0123p0d0000_mp0d0100p0d0153p0d0000.xml",
    )
    parser.add_argument("--keyframe", default="open")
    parser.add_argument(
        "--optimizer",
        choices=["cem", "mjx-autodiff"],
        default="cem",
        help="Optimization strategy for Phase 1 finger controls.",
    )
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--sigma-init", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--contact-distance-threshold", type=float, default=0.01)
    parser.add_argument("--contact-distance-sharpness", type=float, default=0.003)
    parser.add_argument("--score-weight-contact-proxy", type=float, default=0.4)
    parser.add_argument("--score-weight-ctrl-l2", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=240)
    parser.add_argument("--lift-steps", type=int, default=220)
    parser.add_argument("--hold-steps", type=int, default=140)
    parser.add_argument("--lift-delta-z", type=float, default=0.05)
    parser.add_argument(
        "--skip-gif",
        action="store_true",
        help="Skip rollout GIF generation (useful in headless SSH environments).",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "phase1")
    parser.add_argument("--tag", default=None)
    return parser


def _write_history_csv(history: list[dict[str, float]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys()) if history else ["iteration"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _plot_history(history: list[dict[str, float]], out_dir: Path) -> None:
    it = np.array([h["iteration"] for h in history], dtype=np.float64)
    best = np.array([h["iter_best_score"] for h in history], dtype=np.float64)
    mean = np.array([h["iter_mean_score"] for h in history], dtype=np.float64)
    best_so_far = np.array([h["best_score_so_far"] for h in history], dtype=np.float64)
    lift_so_far = np.array([h["best_cube_lift_so_far"] for h in history], dtype=np.float64)
    contacts_so_far = np.array([h["best_contacts_so_far"] for h in history], dtype=np.float64)

    plt.figure(figsize=(8, 4.5))
    plt.plot(it, best, label="iter_best")
    plt.plot(it, mean, label="iter_mean")
    plt.plot(it, best_so_far, label="best_so_far", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.title("Phase 1 Optimization Objective Trace")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "objective_trace.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(it, lift_so_far, label="best cube lift")
    plt.plot(it, contacts_so_far, label="best tip contacts")
    plt.xlabel("Iteration")
    plt.ylabel("Metric")
    plt.title("Best Grasp Metrics Over Optimization")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "grasp_metrics_trace.png", dpi=180)
    plt.close()


def _plot_rollout(cube_z: np.ndarray, contacts: np.ndarray, out_dir: Path) -> None:
    steps = np.arange(cube_z.shape[0], dtype=np.int32)

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, cube_z)
    plt.xlabel("Simulation step")
    plt.ylabel("Cube z")
    plt.title("Best Rollout: Cube Height")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "best_rollout_cube_z.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, contacts)
    plt.xlabel("Simulation step")
    plt.ylabel("Cube-tip contact count")
    plt.title("Best Rollout: Contact Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "best_rollout_contacts.png", dpi=180)
    plt.close()


def _write_report(out_dir: Path, summary: dict, best_metrics: dict[str, float], gif_name: str | None) -> None:
    report_path = out_dir / "report.md"
    lines = [
        "# Phase 1 Inner-Loop Grasp Report",
        "",
        f"- scene_xml: {summary['scene_xml']}",
        f"- keyframe: {summary['keyframe']}",
        f"- best_score: {summary['best_score']:.6f}",
        f"- best_cube_lift: {best_metrics.get('cube_lift', 0.0):.6f}",
        f"- best_contacts: {best_metrics.get('cube_tip_contacts', 0.0):.1f}",
        f"- optimization_wall_time_seconds: {summary.get('timing', {}).get('optimization_wall_time_seconds', 0.0):.3f}",
        f"- mean_iteration_seconds: {summary.get('timing', {}).get('mean_iteration_seconds', 0.0):.3f}",
        f"- rollout_wall_time_seconds: {summary.get('timing', {}).get('rollout_wall_time_seconds', 0.0):.3f}",
        f"- gif_render_wall_time_seconds: {summary.get('timing', {}).get('gif_render_wall_time_seconds', 0.0):.3f}",
        f"- total_run_wall_time_seconds: {summary.get('timing', {}).get('total_run_wall_time_seconds', 0.0):.3f}",
        "",
        "## Artifacts",
        "",
        "- Objective trace: objective_trace.png",
        "- Grasp metrics trace: grasp_metrics_trace.png",
        "- Cube z trajectory: best_rollout_cube_z.png",
        "- Contact trajectory: best_rollout_contacts.png",
        f"- Rollout animation: {gif_name if gif_name is not None else 'skipped'}",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    run_start = time.perf_counter()

    tag = args.tag or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = args.output_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_cfg = Phase1EvalConfig(
        settle_steps=args.settle_steps,
        lift_steps=args.lift_steps,
        hold_steps=args.hold_steps,
        lift_delta_z=args.lift_delta_z,
    )
    opt_cfg = Phase1OptimizationConfig(
        iterations=args.iterations,
        population=args.population,
        elite_fraction=args.elite_fraction,
        sigma_init=args.sigma_init,
        seed=args.seed,
    )
    autodiff_cfg = Phase1AutodiffConfig(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        grad_clip_norm=args.grad_clip_norm,
        contact_distance_threshold=args.contact_distance_threshold,
        contact_distance_sharpness=args.contact_distance_sharpness,
        score_weight_contact_proxy=args.score_weight_contact_proxy,
        score_weight_ctrl_l2=args.score_weight_ctrl_l2,
        seed=args.seed,
    )

    evaluator = Phase1GraspEvaluator(scene_xml=args.scene_xml, keyframe=args.keyframe, cfg=eval_cfg)

    if args.optimizer == "mjx-autodiff":
        result = optimize_finger_controls_autodiff(evaluator=evaluator, cfg=autodiff_cfg)
        optimizer_config = vars(autodiff_cfg)
    else:
        result = optimize_finger_controls(evaluator=evaluator, cfg=opt_cfg)
        optimizer_config = vars(opt_cfg)
    best_ctrl = np.asarray(result["best_finger_ctrl"], dtype=np.float64)
    best_metrics = dict(result["best_metrics"])
    history = list(result["history"])

    rollout_start = time.perf_counter()
    rollout = evaluator.rollout(best_ctrl)
    rollout_seconds = float(time.perf_counter() - rollout_start)
    gif_path: Path | None = None
    gif_render_seconds = 0.0
    if not args.skip_gif:
        try:
            gif_start = time.perf_counter()
            gif_path = evaluator.render_rollout_gif(best_ctrl, out_dir / "best_rollout.gif")
            gif_render_seconds = float(time.perf_counter() - gif_start)
        except Exception as exc:  # pragma: no cover - environment dependent rendering
            print(f"GIF render skipped due to runtime error: {exc}")

    total_run_seconds = float(time.perf_counter() - run_start)

    np.savez_compressed(
        out_dir / "best_rollout.npz",
        qpos=rollout["qpos"],
        qvel=rollout["qvel"],
        cube_z=rollout["cube_z"],
        contacts=rollout["contacts"],
        best_finger_ctrl=best_ctrl,
    )

    summary = {
        "scene_xml": str(args.scene_xml),
        "keyframe": args.keyframe,
        "best_score": float(result["best_score"]),
        "best_finger_ctrl": best_ctrl.tolist(),
        "best_metrics": best_metrics,
        "config_eval": vars(eval_cfg),
        "optimizer": args.optimizer,
        "config_opt": optimizer_config,
        "timing": {
            "optimization_wall_time_seconds": float(result.get("optimization_wall_time_seconds", 0.0)),
            "mean_iteration_seconds": float(result.get("mean_iteration_seconds", 0.0)),
            "rollout_wall_time_seconds": rollout_seconds,
            "gif_render_wall_time_seconds": gif_render_seconds,
            "total_run_wall_time_seconds": total_run_seconds,
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_history_csv(history, out_dir / "optimization_trace.csv")
    _plot_history(history, out_dir)
    _plot_rollout(rollout["cube_z"], rollout["contacts"], out_dir)
    _write_report(out_dir, summary, best_metrics, gif_path.name if gif_path is not None else None)

    print(f"Phase 1 run complete: {out_dir}")
    print(f"Best score: {result['best_score']:.6f}")
    print(f"Best cube lift: {best_metrics.get('cube_lift', 0.0):.6f}")
    print(f"Best contacts: {best_metrics.get('cube_tip_contacts', 0.0):.1f}")


if __name__ == "__main__":
    main()
