from __future__ import annotations

from morphohand.backends.mjx_native import MJXNativeBackend


class MJWarpBackend(MJXNativeBackend):
    """MJWarp backend placeholder using common interface."""

    name = "mjwarp"
    supports_autodiff = False
