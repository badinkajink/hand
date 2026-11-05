from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InnerLoopConfig:
    steps: int = 64
    learning_rate: float = 0.03
    clip: float = 0.15


def optimize_grasp_controls(initial_q: np.ndarray, cfg: InnerLoopConfig) -> np.ndarray:
    """Simple gradient-free placeholder update loop.

    This will be replaced by backend-autodiff gradients once MJX and DiffMJX wiring is added.
    """
    q = initial_q.astype(np.float32).copy()
    for _ in range(cfg.steps):
        direction = -np.sign(q)
        q += cfg.learning_rate * direction
        q = np.clip(q, -cfg.clip, cfg.clip)
    return q
