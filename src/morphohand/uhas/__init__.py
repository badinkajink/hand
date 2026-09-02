"""MorphoHand <-> UHAS (Unified Hand Action Space) interoperability.

UHAS (Casas et al., 2026) represents a hand action as a deformation of a canonical
sphere and maps it to joint commands with Cascade Inverse Kinematics, so ONE policy can
drive hands with different kinematics. Its representation builder consumes a URDF; our
hands are MJCF. This package bridges the two.
"""

from morphohand.uhas.mjcf_to_urdf import HandUrdfExport, export_hand_to_urdf

__all__ = ["HandUrdfExport", "export_hand_to_urdf"]
