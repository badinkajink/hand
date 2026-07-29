from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPaths:
    """Centralized path helpers for model assets."""

    project_root: Path

    @property
    def hand_xml(self) -> Path:
        return self.project_root / "assets" / "mjcf" / "baseline" / "hand.xml"

    @property
    def scene_xml(self) -> Path:
        return self.project_root / "assets" / "mjcf" / "baseline" / "scenes" / "scene.xml"
