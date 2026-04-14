from __future__ import annotations
# pyright: reportMissingImports=false

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .phase1_common import Phase1GraspEvaluator


@dataclass
class Phase1OptimizationConfig:
    iterations: int = 24
    population: int = 40
    elite_fraction: float = 0.2
    sigma_init: float = 0.20
    seed: int = 0


def optimize_finger_controls(
    evaluator: Phase1GraspEvaluator,
    cfg: Phase1OptimizationConfig,
    initial_finger_ctrl: np.ndarray | None = None,
) -> dict[str, Any]:
    """Cross-entropy search for a high-quality grasp control vector."""
    rng = np.random.default_rng(cfg.seed)

    if initial_finger_ctrl is None:
        mean = evaluator.initial_ctrl[evaluator.finger_actuator_ids].astype(np.float64)
    else:
        mean = np.asarray(initial_finger_ctrl, dtype=np.float64)

    sigma = np.full_like(mean, cfg.sigma_init, dtype=np.float64)
    lo, hi = evaluator.finger_ctrl_min, evaluator.finger_ctrl_max

    elite_count = max(2, int(cfg.population * cfg.elite_fraction))

    best_score = -np.inf
    best_ctrl = mean.copy()
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
            s, m = evaluator.evaluate(samples[i])
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
            best_ctrl = samples[iter_best_idx].copy()
            best_metrics = metrics_list[iter_best_idx]

        history.append(
            {
                "iteration": float(it),
                "iter_best_score": iter_best,
                "iter_mean_score": iter_mean,
                "elite_mean_score": float(np.mean(elite_scores)),
                "best_score_so_far": float(best_score),
                "best_cube_lift_so_far": float(best_metrics.get("cube_lift", 0.0)),
                "best_contacts_so_far": float(best_metrics.get("cube_tip_contacts", 0.0)),
                "mean_sigma": float(np.mean(sigma)),
                "iteration_seconds": float(time.perf_counter() - iteration_start),
            }
        )

    optimization_seconds = float(time.perf_counter() - optimization_start)
    mean_iteration_seconds = (
        float(np.mean([row["iteration_seconds"] for row in history])) if history else 0.0
    )

    return {
        "best_finger_ctrl": best_ctrl,
        "best_score": float(best_score),
        "best_metrics": best_metrics,
        "history": history,
        "optimization_wall_time_seconds": optimization_seconds,
        "mean_iteration_seconds": mean_iteration_seconds,
    }
