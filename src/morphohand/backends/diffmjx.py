from __future__ import annotations

from pathlib import Path

import numpy as np

from morphohand.backends.base import RolloutResult
from morphohand.backends.mjx_native import MJXNativeBackend


class DiffMJXLiteBackend(MJXNativeBackend):
    """DiffMJX planning backend.

    Intended for incremental implementation of:
    1) smooth collision branch blending,
    2) contacts-from-distance (CFD) in backward pass,
    3) optional adaptive integration.

    This class currently behaves like MJX native with metadata flags.
    """

    name = "diffmjx-lite"
    supports_autodiff = True

    def __init__(self, enable_cfd: bool = False, enable_smooth_collision: bool = False) -> None:
        super().__init__()
        self.enable_cfd = enable_cfd
        self.enable_smooth_collision = enable_smooth_collision

    def load_model(self, xml_path: Path) -> None:
        super().load_model(xml_path)

    def rollout(self, controls: np.ndarray) -> RolloutResult:
        result = super().rollout(controls)
        result.info["enable_cfd"] = self.enable_cfd
        result.info["enable_smooth_collision"] = self.enable_smooth_collision
        return result

    def metrics(self) -> dict:
        info = super().metrics()
        info["enable_cfd"] = self.enable_cfd
        info["enable_smooth_collision"] = self.enable_smooth_collision
        return info
