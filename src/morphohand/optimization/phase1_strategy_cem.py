from __future__ import annotations
# pyright: reportMissingImports=false

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np

from .phase1_common import Phase1GraspEvaluator


@dataclass
class Phase1OptimizationConfig:
    iterations: int = 24
    population: int = 40
    elite_fraction: float = 0.2
    sigma_init: float = 0.20
    seed: int = 0
    log_every: int = 1


def _run_cem(
    *,
    cfg: Phase1OptimizationConfig,
    mean: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    evaluate_sample: Callable[[np.ndarray], tuple[float, dict[str, float]]],
    log_label: str,
) -> dict[str, Any]:
    """Cross-entropy loop: sample, score, fit elite Gaussian, repeat.

    `evaluate_sample` consumes a flattened sample (already clipped to [lo, hi])
    and returns (score, metrics). The caller owns reshape/interpretation.
    """
    rng = np.random.default_rng(cfg.seed)
    sigma = np.full_like(mean, cfg.sigma_init, dtype=np.float64)
    elite_count = max(2, int(cfg.population * cfg.elite_fraction))

    best_score = -np.inf
    best_sample = mean.copy()
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []
    optimization_start = time.perf_counter()

    for it in range(cfg.iterations):
        iteration_start = time.perf_counter()
        samples = rng.normal(loc=mean, scale=sigma, size=(cfg.population, mean.size))
        samples = np.clip(samples, lo, hi)
        samples[0] = np.clip(mean, lo, hi)

        scores = np.zeros(cfg.population, dtype=np.float64)
        metrics_list: list[dict[str, float]] = []
        for i in range(cfg.population):
            s, m = evaluate_sample(samples[i])
            scores[i] = s
            metrics_list.append(m)

        elite_idx = np.argsort(scores)[-elite_count:]
        elite = samples[elite_idx]
        elite_scores = scores[elite_idx]
        mean = np.mean(elite, axis=0)
        sigma = np.std(elite, axis=0) + 1e-4

        iter_best_idx = int(np.argmax(scores))
        iter_best = float(scores[iter_best_idx])
        iter_mean = float(np.mean(scores))
        if iter_best > best_score:
            best_score = iter_best
            best_sample = samples[iter_best_idx].copy()
            best_metrics = metrics_list[iter_best_idx]

        history.append({
            "iteration": float(it),
            "iter_best_score": iter_best,
            "iter_mean_score": iter_mean,
            "elite_mean_score": float(np.mean(elite_scores)),
            "best_score_so_far": float(best_score),
            "best_cube_lift_so_far": float(best_metrics.get("cube_lift", 0.0)),
            "best_contacts_so_far": float(best_metrics.get("cube_tip_contacts", 0.0)),
            "mean_sigma": float(np.mean(sigma)),
            "iteration_seconds": float(time.perf_counter() - iteration_start),
        })

        if cfg.log_every > 0 and ((it + 1) % cfg.log_every == 0 or it == cfg.iterations - 1):
            elapsed = float(time.perf_counter() - optimization_start)
            eta = float((cfg.iterations - (it + 1)) * (elapsed / max(1, it + 1)))
            print(
                f"[{log_label}] "
                f"iter={it + 1}/{cfg.iterations} "
                f"iter_best={iter_best:.4f} "
                f"best_so_far={best_score:.4f} "
                f"iter_s={history[-1]['iteration_seconds']:.2f} "
                f"eta_s={eta:.1f}"
            )

    optimization_seconds = float(time.perf_counter() - optimization_start)
    mean_iteration_seconds = (
        float(np.mean([row["iteration_seconds"] for row in history])) if history else 0.0
    )
    return {
        "best_sample": best_sample,
        "best_score": float(best_score),
        "best_metrics": best_metrics,
        "history": history,
        "optimization_wall_time_seconds": optimization_seconds,
        "mean_iteration_seconds": mean_iteration_seconds,
    }


def optimize_finger_controls(
    evaluator: Phase1GraspEvaluator,
    cfg: Phase1OptimizationConfig,
    initial_finger_ctrl: np.ndarray | None = None,
) -> dict[str, Any]:
    """Cross-entropy search for a high-quality grasp control vector."""
    if initial_finger_ctrl is None:
        mean = evaluator.initial_ctrl[evaluator.finger_actuator_ids].astype(np.float64)
    else:
        mean = np.asarray(initial_finger_ctrl, dtype=np.float64)

    result = _run_cem(
        cfg=cfg,
        mean=mean,
        lo=evaluator.finger_ctrl_min,
        hi=evaluator.finger_ctrl_max,
        evaluate_sample=evaluator.evaluate,
        log_label="CEM",
    )
    return {
        "best_finger_ctrl": result["best_sample"],
        "best_score": result["best_score"],
        "best_metrics": result["best_metrics"],
        "history": result["history"],
        "optimization_wall_time_seconds": result["optimization_wall_time_seconds"],
        "mean_iteration_seconds": result["mean_iteration_seconds"],
    }


def optimize_finger_control_trajectory(
    evaluator: Phase1GraspEvaluator,
    cfg: Phase1OptimizationConfig,
    phases: int = 4,
    initial_finger_ctrl_traj: np.ndarray | None = None,
) -> dict[str, Any]:
    """CEM over a piecewise-linear finger-control trajectory of `phases` keypoints."""
    n_f = evaluator.finger_actuator_ids.size

    if initial_finger_ctrl_traj is None:
        base = evaluator.initial_ctrl[evaluator.finger_actuator_ids].astype(np.float64)
        mean = np.tile(base, phases)
    else:
        mean = np.asarray(initial_finger_ctrl_traj, dtype=np.float64).reshape(-1)

    lo = np.tile(evaluator.finger_ctrl_min, phases)
    hi = np.tile(evaluator.finger_ctrl_max, phases)

    def evaluate_sample(sample: np.ndarray) -> tuple[float, dict[str, float]]:
        return evaluator.evaluate_trajectory(sample.reshape(phases, n_f))

    result = _run_cem(
        cfg=cfg,
        mean=mean,
        lo=lo,
        hi=hi,
        evaluate_sample=evaluate_sample,
        log_label="CEM-TRAJ",
    )
    return {
        "best_finger_ctrl_traj": result["best_sample"].reshape(phases, n_f),
        "best_score": result["best_score"],
        "best_metrics": result["best_metrics"],
        "history": result["history"],
        "optimization_wall_time_seconds": result["optimization_wall_time_seconds"],
        "mean_iteration_seconds": result["mean_iteration_seconds"],
    }
