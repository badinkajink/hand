from __future__ import annotations
# pyright: reportMissingImports=false

import time
from typing import Literal

import numpy as np

from morphohand.optimization.phase1_grasp import (
    Phase1GraspEvaluator,
    Phase1OptimizationConfig,
    optimize_finger_controls,
)


AdaptationMode = Literal[
    "none",
    "interval-open",
    "interval-initial-fp",
    "sparse-per-morph",
    "local-perturbation",
]


def local_perturb_fp(
    evaluator: Phase1GraspEvaluator,
    baseline_ctrl: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, float]:
    """Sweep +/- delta on each ctrl dim; return the best-scoring ctrl and its score."""
    best_ctrl = baseline_ctrl.copy()
    best_score, _ = evaluator.evaluate(baseline_ctrl)
    for dim in range(len(baseline_ctrl)):
        for sign in (+1, -1):
            candidate = baseline_ctrl.copy()
            candidate[dim] += sign * delta
            score, _ = evaluator.evaluate(candidate)
            if score > best_score:
                best_score = score
                best_ctrl = candidate.copy()
    return best_ctrl, best_score


def adapt_foundational_ctrl(
    mode: AdaptationMode,
    candidate_idx: int,
    interval_ctrl: np.ndarray,
    evaluator: Phase1GraspEvaluator,
    adapt_cfg: Phase1OptimizationConfig,
    refresh_interval: int,
) -> tuple[np.ndarray, bool, float, str]:
    """Return (ctrl, triggered, elapsed_seconds, source_label) per adaptation mode.

    Modes
    -----
    - none: return `interval_ctrl` unchanged.
    - interval-open: every `refresh_interval` morphologies, CEM from scratch.
    - interval-initial-fp: every `refresh_interval`, CEM initialized from `interval_ctrl`.
    - sparse-per-morph: CEM every morphology, initialized from `interval_ctrl`.
    - local-perturbation: +/- sigma_init local search every morphology.
    """
    if mode == "none":
        return interval_ctrl, False, 0.0, "none"

    if mode in {"interval-open", "interval-initial-fp"}:
        should_refresh = (candidate_idx % max(1, refresh_interval)) == 0
    elif mode in {"sparse-per-morph", "local-perturbation"}:
        should_refresh = True
    else:
        raise ValueError(f"Unknown adaptation mode: {mode}")

    if not should_refresh:
        return interval_ctrl, False, 0.0, "reuse"

    if mode == "local-perturbation":
        t0 = time.perf_counter()
        ctrl, _ = local_perturb_fp(evaluator, interval_ctrl, adapt_cfg.sigma_init)
        return ctrl, True, float(time.perf_counter() - t0), "local_perturb"

    if mode == "interval-open":
        init_ctrl: np.ndarray | None = None
        source = "open"
    elif mode == "interval-initial-fp":
        init_ctrl = interval_ctrl
        source = "interval_fp"
    else:
        init_ctrl = interval_ctrl
        source = "sparse"

    t0 = time.perf_counter()
    refined = optimize_finger_controls(
        evaluator=evaluator, cfg=adapt_cfg, initial_finger_ctrl=init_ctrl
    )
    elapsed = float(time.perf_counter() - t0)
    return np.asarray(refined["best_finger_ctrl"], dtype=np.float64), True, elapsed, source
