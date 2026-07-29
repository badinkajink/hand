"""Canonical output paths for rendered videos.

The video tree is `docs/rl/videos/<YYYYMMDD>_<experiment>/<HHMM>_<name>.<ext>` so that plain
alphabetical order is chronological order, both across experiments and within one. Every script
that writes a clip should route through here rather than hardcoding a path, or the tree drifts
back into one flat directory of undated names (which is what
`scripts/reorganize_videos.py` had to undo).

Temp/working renders go to `logs/video_tmp/` — gitignored, per the workspace-layout rule that
run artifacts live under `logs/` and never in the tracked docs tree.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VIDEOS = ROOT / "docs/rl/videos"
TMP = ROOT / "logs/video_tmp"


def experiment_dir(experiment: str, when: datetime | None = None) -> Path:
    """`docs/rl/videos/<YYYYMMDD>_<experiment>/`, created."""
    when = when or datetime.now()
    d = VIDEOS / f"{when.strftime('%Y%m%d')}_{experiment}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def video_out(name: str, experiment: str, ext: str = ".mp4", when: datetime | None = None) -> Path:
    """Full path for a clip: `<YYYYMMDD>_<experiment>/<HHMM>_<name><ext>`."""
    when = when or datetime.now()
    if not name.endswith(ext):
        name = name + ext
    return experiment_dir(experiment, when) / f"{when.strftime('%H%M')}_{name}"


def sidecar(video: Path, suffix: str) -> Path:
    """Path for a file that must travel WITH a video (e.g. `.health.json`).

    Keeps the video's timestamp prefix so the reorganizer pairs them.
    """
    return video.with_suffix("").with_suffix(suffix) if video.suffix else video.with_suffix(suffix)


def tmp_dir(name: str) -> Path:
    """Gitignored scratch dir for intermediate renders."""
    d = TMP / name
    d.mkdir(parents=True, exist_ok=True)
    return d
