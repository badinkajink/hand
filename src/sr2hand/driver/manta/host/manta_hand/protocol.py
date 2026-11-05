"""Wire-format helpers shared by driver.py and joint.py.

Mirrors firmware/Core/Src/protocol.c and docs/protocol.md -- keep both in
sync if the command set changes.
"""

NUM_JOINTS = 8


class MantaHandError(Exception):
    """Raised when the firmware replies with ERR, or the link misbehaves."""


def format_command(*parts) -> str:
    return " ".join(str(p) for p in parts)
