"""CEM in a low-dim synergy subspace (eigengrasp coefficients).

Mirrors ``phase1_strategy_cem`` but samples in K-dim coefficient space and
projects back to the 9D finger control vector (or `(phases, 9)` trajectory)
before calling the evaluator. The subspace is supplied as a ``SynergyBasis``.

Why: search dimension drops from 9 (static) or 36 (4-phase trajectory) to K or
K*phases, which lets CEM converge with smaller populations and ought to
sample only "natural" grasp directions.
"""

from __future__ import annotations
# pyright: reportMissingImports=false

from typing import Any

import numpy as np

from .eigengrasp import SynergyBasis, bounds_in_subspace
from .phase1_common import Phase1GraspEvaluator
from .phase1_strategy_cem import Phase1OptimizationConfig, _run_cem


def optimize_finger_controls_synergy(
    evaluator: Phase1GraspEvaluator,
    cfg: Phase1OptimizationConfig,
    basis: SynergyBasis,
    initial_finger_ctrl: np.ndarray | None = None,
) -> dict[str, Any]:
    """CEM in the K-dim synergy space for a static grasp."""
    if basis.n_full != evaluator.finger_actuator_ids.size:
        raise ValueError(
            f"basis.n_full ({basis.n_full}) does not match finger ctrl dim "
            f"({evaluator.finger_actuator_ids.size})"
        )

    if initial_finger_ctrl is None:
        init_full = evaluator.initial_ctrl[evaluator.finger_actuator_ids].astype(np.float64)
    else:
        init_full = np.asarray(initial_finger_ctrl, dtype=np.float64)
    mean_sub = basis.project(init_full)

    lo_sub, hi_sub = bounds_in_subspace(
        basis,
        evaluator.finger_ctrl_min,
        evaluator.finger_ctrl_max,
    )

    lo_full = evaluator.finger_ctrl_min
    hi_full = evaluator.finger_ctrl_max

    def evaluate_sample(sample_sub: np.ndarray) -> tuple[float, dict[str, float]]:
        full = basis.reconstruct(sample_sub)
        full = np.clip(full, lo_full, hi_full)
        return evaluator.evaluate(full)

    result = _run_cem(
        cfg=cfg,
        mean=mean_sub,
        lo=lo_sub,
        hi=hi_sub,
        evaluate_sample=evaluate_sample,
        log_label="CEM-SYN",
    )

    best_full = np.clip(basis.reconstruct(result["best_sample"]), lo_full, hi_full)
    return {
        "best_finger_ctrl": best_full,
        "best_sub_coeffs": result["best_sample"],
        "best_score": result["best_score"],
        "best_metrics": result["best_metrics"],
        "history": result["history"],
        "optimization_wall_time_seconds": result["optimization_wall_time_seconds"],
        "mean_iteration_seconds": result["mean_iteration_seconds"],
    }


def optimize_finger_control_trajectory_synergy(
    evaluator: Phase1GraspEvaluator,
    cfg: Phase1OptimizationConfig,
    basis: SynergyBasis,
    phases: int = 4,
    initial_finger_ctrl_traj: np.ndarray | None = None,
) -> dict[str, Any]:
    """CEM over a per-phase synergy coefficient trajectory.

    The full optimization vector is (phases * K). Each phase reconstructs to a
    9D finger control vector via the basis; the existing trajectory
    interpolator handles between-phase blending.
    """
    n_f = int(evaluator.finger_actuator_ids.size)
    if basis.n_full != n_f:
        raise ValueError(f"basis.n_full ({basis.n_full}) does not match finger ctrl dim ({n_f})")
    K = basis.n_components

    if initial_finger_ctrl_traj is None:
        base = evaluator.initial_ctrl[evaluator.finger_actuator_ids].astype(np.float64)
        init_full_traj = np.tile(base, (phases, 1))
    else:
        init_full_traj = np.asarray(initial_finger_ctrl_traj, dtype=np.float64).reshape(phases, n_f)

    init_sub = np.stack([basis.project(init_full_traj[p]) for p in range(phases)], axis=0)
    mean = init_sub.reshape(-1)

    lo_sub, hi_sub = bounds_in_subspace(basis, evaluator.finger_ctrl_min, evaluator.finger_ctrl_max)
    lo = np.tile(lo_sub, phases)
    hi = np.tile(hi_sub, phases)

    lo_full = evaluator.finger_ctrl_min
    hi_full = evaluator.finger_ctrl_max

    def _expand_to_full_traj(sample: np.ndarray) -> np.ndarray:
        sub = sample.reshape(phases, K)
        full = np.stack([basis.reconstruct(sub[p]) for p in range(phases)], axis=0)
        return np.clip(full, lo_full, hi_full)

    def evaluate_sample(sample: np.ndarray) -> tuple[float, dict[str, float]]:
        full_traj = _expand_to_full_traj(sample)
        return evaluator.evaluate_trajectory(full_traj)

    result = _run_cem(
        cfg=cfg,
        mean=mean,
        lo=lo,
        hi=hi,
        evaluate_sample=evaluate_sample,
        log_label="CEM-SYN-TRAJ",
    )

    best_full_traj = _expand_to_full_traj(result["best_sample"])
    return {
        "best_finger_ctrl_traj": best_full_traj,
        "best_sub_coeffs_traj": result["best_sample"].reshape(phases, K),
        "best_score": result["best_score"],
        "best_metrics": result["best_metrics"],
        "history": result["history"],
        "optimization_wall_time_seconds": result["optimization_wall_time_seconds"],
        "mean_iteration_seconds": result["mean_iteration_seconds"],
    }
