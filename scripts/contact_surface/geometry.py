"""The existing hom_chain cap formula, with its two area conventions made explicit."""

from dataclasses import asdict, dataclass
import math

import numpy as np

from hom_contact_rig import fib_cap


@dataclass(frozen=True)
class Surface:
    spacing_mm: float = 1.0
    radius: float = 0.01055
    sphere_radius: float = 0.00075
    cap_degrees: float = 45.0
    modulus: float = 1e7
    relaxation: float = 0.02
    nominal_inverse_inertia: float = 47.21483925057961

    def parameters(self):
        if not (
            self.spacing_mm > 0
            and 0 < self.sphere_radius < self.radius
            and 0 < self.cap_degrees <= 90
            and self.modulus > 0
            and self.relaxation > 0
            and self.nominal_inverse_inertia > 0
        ):
            raise ValueError("Invalid surface/material parameters")
        cap = math.radians(self.cap_degrees)
        area = 2 * math.pi * self.radius**2 * (1 - math.cos(cap))
        count = max(1, round(area / (self.spacing_mm * 1e-3) ** 2))
        correction = ((self.radius - self.sphere_radius) / self.radius) ** 2
        nominal_area = area / count
        stiffness = self.modulus / self.radius * nominal_area * correction
        tc = self.relaxation / 2
        impedance = 1 - 1 / (tc**2 * stiffness * self.nominal_inverse_inertia)
        if not 0.0001 <= impedance <= 0.9999:
            raise ValueError(
                f"Unrepresentable impedance {impedance:g}; do not silently clamp/retune"
            )
        return dict(
            **asdict(self),
            count=count,
            cap_area=area,
            nominal_area=nominal_area,
            center_radius_correction=correction,
            corrected_area=nominal_area * correction,
            stiffness=stiffness,
            solref=[tc, 1.0],
            solimp=[impedance, impedance, 0.001, 0.5, 2.0],
        )

    def centers(self, phase_degrees=0):
        p = self.parameters()
        directions = fib_cap(p["count"], math.radians(self.cap_degrees))
        a = math.radians(phase_degrees)
        rotation = np.array(
            [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        )
        # Match hom_chain's MJCF coordinate precision.
        return np.round((self.radius - self.sphere_radius) * directions @ rotation.T, 7)
