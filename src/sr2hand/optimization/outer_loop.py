from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MapElitesConfig:
    population: int = 32
    generations: int = 20
    mutation_sigma: float = 0.05


def initialize_morphology_population(population: int, dof: int = 9) -> np.ndarray:
    """Initialize morphology vectors for (x,y,l) across 3 fingers."""
    return np.random.uniform(low=-0.05, high=0.05, size=(population, dof)).astype(np.float32)
