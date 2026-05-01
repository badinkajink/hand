from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FeasibilityCriteria:
    max_mean_tip_distance: float = float("inf")
    min_contacts: float = 0.0
    min_finger_contact_persistence: float | None = None
    max_finger_yaw_drift: float | None = None
    max_cube_xy_drift: float | None = None
    max_cube_yaw_drift: float | None = None
    max_cube_axis_tilt: float | None = None
    max_cube_ang_drift: float | None = None


def is_feasible(metrics: dict[str, float], criteria: FeasibilityCriteria) -> bool:
    if float(metrics.get("mean_tip_distance", np.inf)) > criteria.max_mean_tip_distance:
        return False
    if float(metrics.get("cube_tip_contacts", 0.0)) < criteria.min_contacts:
        return False
    if criteria.min_finger_contact_persistence is not None:
        if (
            float(metrics.get("min_finger_contact_persistence", 0.0))
            < criteria.min_finger_contact_persistence
        ):
            return False
    if criteria.max_finger_yaw_drift is not None:
        if float(metrics.get("finger_yaw_drift", np.inf)) > criteria.max_finger_yaw_drift:
            return False
    if criteria.max_cube_xy_drift is not None:
        if float(metrics.get("cube_xy_drift", np.inf)) > criteria.max_cube_xy_drift:
            return False
    if criteria.max_cube_yaw_drift is not None:
        if float(metrics.get("cube_yaw_drift", np.inf)) > criteria.max_cube_yaw_drift:
            return False
    if criteria.max_cube_axis_tilt is not None:
        if float(metrics.get("cube_axis_tilt", np.inf)) > criteria.max_cube_axis_tilt:
            return False
    if criteria.max_cube_ang_drift is not None:
        if float(metrics.get("cube_ang_drift", np.inf)) > criteria.max_cube_ang_drift:
            return False
    return True


PARETO_MAX_KEYS: tuple[str, ...] = ("score", "cube_lift", "cube_tip_contacts")
PARETO_MIN_KEYS: tuple[str, ...] = ("mean_tip_distance", "cube_vel_norm")


def pareto_front_indices(
    rows: list[dict[str, object]],
    max_keys: Iterable[str] = PARETO_MAX_KEYS,
    min_keys: Iterable[str] = PARETO_MIN_KEYS,
) -> list[int]:
    max_keys_t = tuple(max_keys)
    min_keys_t = tuple(min_keys)
    if not rows:
        return []

    n = len(rows)
    dominated = [False] * n
    for i, a in enumerate(rows):
        if dominated[i]:
            continue
        for j, b in enumerate(rows):
            if i == j or dominated[i]:
                continue
            b_not_worse = all(float(b[k]) >= float(a[k]) for k in max_keys_t) and all(
                float(b[k]) <= float(a[k]) for k in min_keys_t
            )
            b_strict = any(float(b[k]) > float(a[k]) for k in max_keys_t) or any(
                float(b[k]) < float(a[k]) for k in min_keys_t
            )
            if b_not_worse and b_strict:
                dominated[i] = True

    return [i for i, d in enumerate(dominated) if not d]
