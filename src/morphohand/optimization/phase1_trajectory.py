"""Trajectory-based grasp optimization support.

Provides helpers and utilities for optimizing time-varying finger-control trajectories
over the grasp+lift+pivot+hold manipulation sequence.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .phase1_common import Phase1GraspEvaluator


class TrajectoryInterpolator:
    """Interpolate finger control keypoints across manipulation phases."""

    def __init__(
        self,
        trajectory: np.ndarray,
        lift_steps: int,
        pivot_steps: int,
        hold_steps: int,
        lift_ramp_steps: int,
        pivot_ramp_steps: int,
    ):
        """Initialize trajectory interpolator.

        Args:
            trajectory: Shape (phases, n_f). Assumed phases >= 2:
              - Index 0: settle phase
              - Index 1: end of lift
              - Index 2: end of pivot
              - Index 3: end of hold
            lift_steps, pivot_steps, hold_steps: Phase durations.
            lift_ramp_steps, pivot_ramp_steps: Ramp-up durations for smooth control.
        """
        self.trajectory = np.asarray(trajectory, dtype=np.float64)
        self.phases, self.n_f = self.trajectory.shape
        self.lift_steps = int(lift_steps)
        self.pivot_steps = int(pivot_steps)
        self.hold_steps = int(hold_steps)
        self.lift_ramp_steps = int(lift_ramp_steps)
        self.pivot_ramp_steps = int(pivot_ramp_steps)

    def at_dynamic_step(self, dynamic_t: int) -> np.ndarray:
        """Get interpolated finger control at a dynamic phase timestep.

        Args:
            dynamic_t: Step index within the dynamic (non-settle) portion.
                Maps to lift → pivot → hold sequence.

        Returns:
            Interpolated finger control vector of shape (n_f,).
        """
        if self.phases < 2:
            return self.trajectory[0]

        if dynamic_t < 0:
            return self.trajectory[0]

        if dynamic_t < self.lift_steps:
            # Lift phase: interpolate from phase[0] to phase[1]
            denom = max(1, self.lift_steps - 1)
            frac = float(dynamic_t) / denom
            a = self.trajectory[0]
            b = self.trajectory[min(1, self.phases - 1)]
            return (1.0 - frac) * a + frac * b
        elif dynamic_t < (self.lift_steps + self.pivot_steps):
            # Pivot phase: interpolate from phase[1] to phase[2]
            t = dynamic_t - self.lift_steps
            denom = max(1, self.pivot_steps - 1)
            frac = float(t) / denom
            a = self.trajectory[min(1, self.phases - 1)]
            b = self.trajectory[min(2, self.phases - 1)]
            return (1.0 - frac) * a + frac * b
        else:
            # Hold phase: interpolate from phase[2] to phase[3]
            t = dynamic_t - self.lift_steps - self.pivot_steps
            denom = max(1, self.hold_steps - 1)
            frac = float(t) / denom if denom > 0 else 1.0
            a = self.trajectory[min(2, self.phases - 1)]
            b = self.trajectory[min(3, self.phases - 1)]
            return (1.0 - frac) * a + frac * b


def build_trajectory_interpolator(
    trajectory: np.ndarray,
    evaluator: Phase1GraspEvaluator,
) -> TrajectoryInterpolator:
    """Factory to construct a TrajectoryInterpolator from an evaluator config.

    Args:
        trajectory: Shape (phases, n_f).
        evaluator: Phase1GraspEvaluator instance.

    Returns:
        Initialized TrajectoryInterpolator.
    """
    return TrajectoryInterpolator(
        trajectory=trajectory,
        lift_steps=int(evaluator.cfg.lift_steps),
        pivot_steps=int(evaluator.cfg.pivot_steps),
        hold_steps=int(evaluator.cfg.hold_steps),
        lift_ramp_steps=int(evaluator.cfg.lift_ramp_steps),
        pivot_ramp_steps=int(evaluator.cfg.pivot_ramp_steps),
    )
