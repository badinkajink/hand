from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class RolloutResult:
    observations: np.ndarray
    rewards: np.ndarray
    done: np.ndarray
    info: dict


class PhysicsBackend(Protocol):
    """Common interface for simulator backends."""

    name: str
    supports_autodiff: bool

    def load_model(self, xml_path: Path) -> None:
        ...

    def reset(self, seed: int | None = None) -> np.ndarray:
        ...

    def step(self, control: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        ...

    def rollout(self, controls: np.ndarray) -> RolloutResult:
        ...

    def metrics(self) -> dict:
        ...
