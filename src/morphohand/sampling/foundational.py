from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FoundationalPose:
    label: str
    finger_ctrl: np.ndarray
    score: float


def _load_pose_from_summary(summary_path: Path, label: str) -> FoundationalPose:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return FoundationalPose(
        label=label,
        finger_ctrl=np.asarray(payload["best_finger_ctrl"], dtype=np.float64),
        score=float(payload.get("best_score", float("-inf"))),
    )


def load_foundational_poses(
    run_dir: Path,
    keyframe_name: str | None = None,
) -> list[FoundationalPose]:
    """Load FP summaries from subdirs of `run_dir`, sorted by score (desc).

    If `keyframe_name` is given, only include summaries whose `keyframe` field matches.
    """
    poses: list[FoundationalPose] = []
    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir():
            continue
        summary_path = subdir / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if keyframe_name is not None and str(payload.get("keyframe")) != keyframe_name:
            continue
        poses.append(
            FoundationalPose(
                label=subdir.name,
                finger_ctrl=np.asarray(payload["best_finger_ctrl"], dtype=np.float64),
                score=float(payload.get("best_score", float("-inf"))),
            )
        )

    if not poses:
        suffix = f" for keyframe '{keyframe_name}'" if keyframe_name else ""
        raise ValueError(f"No foundational summaries{suffix} under {run_dir}")

    return sorted(poses, key=lambda p: p.score, reverse=True)


def load_foundational_poses_with_scene(
    run_dir: Path,
) -> tuple[list[FoundationalPose], Path]:
    """Like `load_foundational_poses` (no keyframe filter) but also returns the
    unique `scene_xml` referenced by the summaries and validates consistency."""
    poses: list[FoundationalPose] = []
    scene_xml_path: Path | None = None
    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir():
            continue
        summary_path = subdir / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        poses.append(
            FoundationalPose(
                label=subdir.name,
                finger_ctrl=np.asarray(payload["best_finger_ctrl"], dtype=np.float64),
                score=float(payload.get("best_score", float("-inf"))),
            )
        )
        current_scene = Path(payload["scene_xml"])
        if scene_xml_path is None:
            scene_xml_path = current_scene
        elif current_scene != scene_xml_path:
            raise ValueError(
                f"Mixed scene_xml values in {run_dir}; expected a single-scene foundational run."
            )

    if not poses or scene_xml_path is None:
        raise ValueError(f"No foundational summaries under {run_dir}")

    return sorted(poses, key=lambda p: p.score, reverse=True), scene_xml_path


def load_best_pose_with_prefix(
    run_dir: Path,
    label_prefix: str,
    label: str,
) -> FoundationalPose:
    """Scan subdirs whose name starts with `label_prefix`, return the highest-scoring
    summary tagged with the given display `label`."""
    candidates: list[tuple[float, Path]] = []
    for subdir in sorted(run_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith(label_prefix):
            continue
        summary_path = subdir / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        candidates.append((float(payload.get("best_score", float("-inf"))), summary_path))

    if not candidates:
        raise ValueError(f"No subruns matching prefix '{label_prefix}' in {run_dir}")

    _, best_summary = max(candidates, key=lambda item: item[0])
    return _load_pose_from_summary(best_summary, label)
