from __future__ import annotations

from sr2hand.backends.mjx_native import MJXNativeBackend


class MJWarpBackend(MJXNativeBackend):
    """MJWarp backend placeholder using common interface."""

    name = "mjwarp"
    supports_autodiff = False
