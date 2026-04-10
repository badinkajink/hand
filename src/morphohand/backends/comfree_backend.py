from __future__ import annotations

from morphohand.backends.mjwarp_backend import MJWarpBackend


class ComFreeBackend(MJWarpBackend):
    """ComFree Warp backend placeholder using common interface."""

    name = "comfree-warp"
    supports_autodiff = False

    def __init__(self, stiffness: float = 0.2, damping: float = 0.001) -> None:
        super().__init__()
        self.stiffness = stiffness
        self.damping = damping

    def metrics(self) -> dict:
        info = super().metrics()
        info["comfree_stiffness"] = self.stiffness
        info["comfree_damping"] = self.damping
        return info
