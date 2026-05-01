"""Trajectory-based grasp optimization support.

Helpers for optimizing time-varying finger-control trajectories over the
grasp+lift+pivot+hold manipulation sequence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .phase1_common import Phase1GraspEvaluator


class TrajectoryInterpolator:
    """Linearly interpolate finger-control keypoints across the dynamic phases.

    Trajectory shape is `(phases, n_f)`. With phases >= 4 the conventional layout is:
      index 0 = settle/end-of-grasp, 1 = end-of-lift, 2 = end-of-pivot, 3 = end-of-hold.
    Fewer phases collapse to the last available keypoint.
    """

    def __init__(
        self,
        trajectory: np.ndarray,
        lift_steps: int,
        pivot_steps: int,
        hold_steps: int,
    ):
        self.trajectory = np.asarray(trajectory, dtype=np.float64)
        self.phases, self.n_f = self.trajectory.shape
        self.lift_steps = int(lift_steps)
        self.pivot_steps = int(pivot_steps)
        self.hold_steps = int(hold_steps)

    def at_dynamic_step(self, dynamic_t: int) -> np.ndarray:
        """Interpolated finger control at an absolute dynamic-phase step (lift→pivot→hold)."""
        if self.phases < 2 or dynamic_t < 0:
            return self.trajectory[0]

        if dynamic_t < self.lift_steps:
            local_t, local_steps, a_idx, b_idx = dynamic_t, self.lift_steps, 0, 1
        elif dynamic_t < self.lift_steps + self.pivot_steps:
            local_t = dynamic_t - self.lift_steps
            local_steps = self.pivot_steps
            a_idx, b_idx = 1, 2
        else:
            local_t = dynamic_t - self.lift_steps - self.pivot_steps
            local_steps = self.hold_steps
            a_idx, b_idx = 2, 3

        denom = max(1, local_steps - 1)
        frac = float(local_t) / denom if local_steps > 1 else 1.0
        a = self.trajectory[min(a_idx, self.phases - 1)]
        b = self.trajectory[min(b_idx, self.phases - 1)]
        return (1.0 - frac) * a + frac * b


def build_trajectory_interpolator(
    trajectory: np.ndarray,
    evaluator: "Phase1GraspEvaluator",
) -> TrajectoryInterpolator:
    return TrajectoryInterpolator(
        trajectory=trajectory,
        lift_steps=int(evaluator.cfg.lift_steps),
        pivot_steps=int(evaluator.cfg.pivot_steps),
        hold_steps=int(evaluator.cfg.hold_steps),
    )
