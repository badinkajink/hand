from __future__ import annotations

from sr2hand.backends.comfree_backend import ComFreeBackend
from sr2hand.backends.mjwarp_backend import MJWarpBackend
from sr2hand.backends.diffmjx import DiffMJXLiteBackend
from sr2hand.backends.mjx_native import MJXNativeBackend


def create_backend(name: str):
    """Create a backend by stable name."""
    normalized = name.strip().lower()
    registry = {
        "mjx": MJXNativeBackend,
        "mjx-native": MJXNativeBackend,
        "diffmjx": DiffMJXLiteBackend,
        "diffmjx-lite": DiffMJXLiteBackend,
        "mjwarp": MJWarpBackend,
        "comfree": ComFreeBackend,
        "comfree-warp": ComFreeBackend,
    }
    if normalized not in registry:
        options = ", ".join(sorted(registry))
        raise ValueError(f"Unknown backend '{name}'. Available backends: {options}")
    return registry[normalized]()
