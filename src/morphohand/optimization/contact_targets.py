"""Object-frame contact target patches as a grasp specification primitive.

Instead of letting the evaluator only count tip-object collisions, this module
lets the user *specify where* on the object each fingertip should land. A
target patch is a (point, optional normal, radius, optional finger
assignment) tuple expressed in object body-local coordinates — so the patch
moves with the object as it tilts or drifts.

The scorer turns "fingertips should land on these patches" into a smooth
scalar contribution to the objective:

    score_contact = sum_p reward(patch_p, assigned_tip(p))

where the reward decays with distance from the patch center and saturates at
the patch radius. Unassigned patches greedily take whichever remaining tip
is closest (Hungarian-style for 3 fingers x <=3 patches; brute force is fine).

Authoring format (YAML)::

    object_body: power_drill
    patches:
      - name: trigger
        finger: index
        local_pos: [0.02, -0.01, 0.04]
        local_normal: [1.0, 0.0, 0.0]
        radius: 0.012
      - name: barrel_top
        finger: thumb
        local_pos: [-0.02, 0.0, 0.05]
        local_normal: [-1.0, 0.0, 0.5]
        radius: 0.015
      - name: handle_back
        finger: middle
        local_pos: [0.0, 0.02, -0.02]
        local_normal: [0.0, 1.0, 0.0]
        radius: 0.012

This solves "specify contact patches on an object" directly and decouples the
intent ("grip the trigger") from the joint-space control that achieves it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:  # pyyaml is already a project dep
    import yaml  # pyright: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - pyyaml is required at runtime
    yaml = None


_FINGER_NAMES = ("thumb", "index", "middle")
_FINGER_INDEX = {name: i for i, name in enumerate(_FINGER_NAMES)}


@dataclass
class ContactTarget:
    """A single target patch on the object surface, expressed in body-local frame."""

    name: str
    local_pos: np.ndarray
    radius: float = 0.012
    local_normal: np.ndarray | None = None
    finger: str | None = None  # one of "thumb"/"index"/"middle", or None to auto-assign

    def __post_init__(self) -> None:
        self.local_pos = np.asarray(self.local_pos, dtype=np.float64).reshape(3)
        if self.local_normal is not None:
            n = np.asarray(self.local_normal, dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(n))
            if norm < 1e-12:
                raise ValueError(f"target '{self.name}': zero-length normal")
            self.local_normal = n / norm
        if self.finger is not None and self.finger not in _FINGER_INDEX:
            raise ValueError(
                f"target '{self.name}': finger must be one of {_FINGER_NAMES} or None"
            )
        if self.radius <= 0:
            raise ValueError(f"target '{self.name}': radius must be positive")

    @property
    def finger_idx(self) -> int | None:
        return _FINGER_INDEX[self.finger] if self.finger is not None else None


@dataclass
class ContactTargetSet:
    """A bundle of target patches for one object body."""

    object_body: str
    patches: list[ContactTarget] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "ContactTargetSet":
        if yaml is None:
            raise ImportError("pyyaml is required to load contact target YAML files")
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ContactTargetSet":
        patches = [
            ContactTarget(
                name=p["name"],
                local_pos=p["local_pos"],
                radius=float(p.get("radius", 0.012)),
                local_normal=p.get("local_normal"),
                finger=p.get("finger"),
            )
            for p in raw.get("patches", [])
        ]
        return cls(object_body=raw.get("object_body", "object"), patches=patches)


def world_targets(
    target_set: ContactTargetSet,
    object_world_pos: np.ndarray,
    object_world_rot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | list[np.ndarray | None]]:
    """Transform target patch positions/normals to world frame.

    Returns (positions, normals). `normals` is a list with `None` entries
    preserved for patches that didn't specify a normal.
    """
    if not target_set.patches:
        return np.zeros((0, 3), dtype=np.float64), []

    locals_pos = np.stack([p.local_pos for p in target_set.patches], axis=0)
    pos_world = locals_pos @ object_world_rot.T + object_world_pos[None, :]
    normals: list[np.ndarray | None] = []
    for p in target_set.patches:
        if p.local_normal is None:
            normals.append(None)
        else:
            normals.append(object_world_rot @ p.local_normal)
    return pos_world, normals


def _assign(
    n_targets: int,
    fixed_assignments: list[int | None],
    tip_to_patch_distances: np.ndarray,
) -> list[int | None]:
    """Greedy/Hungarian assignment of fingertips to patches.

    With at most 3 fingers and ~3 patches the search space is tiny — we
    enumerate every assignment honoring fixed (finger-tagged) patches.

    Returns: per-patch tip index (or None if unassigned in best solution).
    """
    if n_targets == 0:
        return []

    n_tips = tip_to_patch_distances.shape[0]
    available_tips = [t for t in range(n_tips) if t not in {f for f in fixed_assignments if f is not None}]
    free_idx = [i for i, f in enumerate(fixed_assignments) if f is None]

    best_assignment: list[int | None] = list(fixed_assignments)
    best_cost = float("inf")

    for combo in itertools.permutations(available_tips, min(len(free_idx), len(available_tips))):
        trial: list[int | None] = list(fixed_assignments)
        for slot, tip in zip(free_idx, combo):
            trial[slot] = tip
        cost = 0.0
        for patch_i, tip_i in enumerate(trial):
            if tip_i is None:
                cost += 1.0  # missing assignment penalty
            else:
                cost += float(tip_to_patch_distances[tip_i, patch_i])
        if cost < best_cost:
            best_cost = cost
            best_assignment = trial

    return best_assignment


def _patch_reward(distance: float, radius: float) -> float:
    """Smooth reward: 1 inside the patch, decaying to 0 at ~3 radii."""
    x = distance / max(radius, 1e-9)
    if x <= 1.0:
        return 1.0
    if x >= 3.0:
        return 0.0
    return float(0.5 * (1.0 + np.cos(np.pi * (x - 1.0) / 2.0)))


@dataclass
class ContactTargetScoreBreakdown:
    """Per-patch diagnostics returned alongside the scalar score."""

    assignment: list[int | None]
    distances: list[float]
    rewards: list[float]
    mean_distance: float
    total_reward: float


def score_contact_targets(
    target_set: ContactTargetSet,
    tip_positions_world: np.ndarray,
    object_world_pos: np.ndarray,
    object_world_rot: np.ndarray,
) -> ContactTargetScoreBreakdown:
    """Score how well current fingertip positions match the target patches.

    `tip_positions_world` should be (3, 3) for [thumb, index, middle].
    Returns a breakdown; the scalar to put in the objective is the
    `total_reward` (or `-mean_distance` if you prefer hard distance).
    """
    if not target_set.patches:
        return ContactTargetScoreBreakdown([], [], [], 0.0, 0.0)

    if tip_positions_world.shape != (3, 3):
        raise ValueError(f"expected tip positions (3, 3); got {tip_positions_world.shape}")

    patch_pos_world, _ = world_targets(target_set, object_world_pos, object_world_rot)
    # tip_to_patch[t, p] = ||tip_t - patch_p||
    diffs = tip_positions_world[:, None, :] - patch_pos_world[None, :, :]
    tip_to_patch = np.linalg.norm(diffs, axis=-1)

    fixed = [p.finger_idx for p in target_set.patches]
    assignment = _assign(len(target_set.patches), fixed, tip_to_patch)

    distances: list[float] = []
    rewards: list[float] = []
    for patch_i, tip_i in enumerate(assignment):
        if tip_i is None:
            distances.append(float("inf"))
            rewards.append(0.0)
            continue
        d = float(tip_to_patch[tip_i, patch_i])
        distances.append(d)
        rewards.append(_patch_reward(d, target_set.patches[patch_i].radius))

    finite_d = [d for d in distances if np.isfinite(d)]
    mean_d = float(np.mean(finite_d)) if finite_d else 0.0
    return ContactTargetScoreBreakdown(
        assignment=assignment,
        distances=distances,
        rewards=rewards,
        mean_distance=mean_d,
        total_reward=float(sum(rewards)),
    )


def load_contact_target_set(path: str | Path) -> ContactTargetSet:
    return ContactTargetSet.from_yaml(Path(path))


def patches_as_world_array(
    target_set: ContactTargetSet,
    object_world_pos: np.ndarray,
    object_world_rot: np.ndarray,
) -> np.ndarray:
    """Return (n_patches, 3) array of patch centers in world frame."""
    pos, _ = world_targets(target_set, object_world_pos, object_world_rot)
    return pos


def _ensure_iterable_patches(patches: Iterable[Any]) -> list[ContactTarget]:
    out: list[ContactTarget] = []
    for p in patches:
        if isinstance(p, ContactTarget):
            out.append(p)
        elif isinstance(p, dict):
            out.append(
                ContactTarget(
                    name=p["name"],
                    local_pos=p["local_pos"],
                    radius=float(p.get("radius", 0.012)),
                    local_normal=p.get("local_normal"),
                    finger=p.get("finger"),
                )
            )
        else:
            raise TypeError(f"patch must be ContactTarget or dict, got {type(p)}")
    return out
