"""Reorganize docs/rl/videos into TIMESTAMPED experiment folders, temporally ordered.

Before: one flat `reorient/` holding ~90 clips spanning two months, names carrying no date, so
neither the folder listing nor the filenames tell you what ran when or what belongs together.

After:  docs/rl/videos/<YYYYMMDD>_<experiment>/<HHMM>_<name>.<ext>

Sorting the top level gives chronological order across experiments; sorting inside a folder gives
chronological order within one. The experiment key is (date, group) — this project's log is
chronological and a day's clips ARE one experiment, so grouping by day is the honest boundary
rather than inventing semantic labels after the fact.

Timestamps come from file mtime. Sidecars (`.health.json`, `.png`, `.gif`) that share a stem with
a video move WITH it and take the same prefix, so pairs never separate.

Follows the `rename_results_bids.sh` pattern: dry-run by default, `--apply` to execute, uses
`git mv` so history follows, rewrites path references across tracked text files, and regenerates
`docs/rl/videos/INDEX.md`. Idempotent — a file already carrying its `HHMM_` prefix in the right
folder is left alone.

  uv run python scripts/reorganize_videos.py            # show the plan
  uv run python scripts/reorganize_videos.py --apply
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "docs/rl/videos"
MEDIA_EXT = {".mp4", ".gif"}
SIDECAR_EXT = {".json", ".png", ".txt"}
# Text files whose contents may reference a video path.
TEXT_GLOBS = ("*.md", "*.py", "*.sh", "*.typ", "*.tex", "*.yaml", "*.yml")
SKIP_DIRS = {".git", "external", "results", "wandb", "node_modules", "__pycache__", ".venv"}

STAMP_RE = re.compile(r"^\d{4}_")


def group_of(rel: Path) -> str:
    """Experiment group from the file's current location under docs/rl/videos."""
    parts = rel.parts[:-1]
    if not parts:
        return "misc"
    # reorient/sweep -> sweep ; reorient/sim2real -> sim2real ; reorient -> reorient
    named = [p for p in parts if not p.startswith("_")]
    if not named:
        return "tmp"
    return named[-1]


def plan() -> list[tuple[Path, Path]]:
    """[(old, new)] absolute paths, sidecars following their video's prefix."""
    if not VIDEOS.exists():
        return []
    files = [p for p in VIDEOS.rglob("*") if p.is_file()]

    # stem -> prefix, so a sidecar inherits its video's timestamp instead of its own
    video_prefix: dict[tuple[str, str], tuple[str, str]] = {}
    for p in sorted(files):
        if p.suffix.lower() not in MEDIA_EXT:
            continue
        rel = p.relative_to(VIDEOS)
        ts = datetime.fromtimestamp(p.stat().st_mtime)
        stem = STAMP_RE.sub("", p.stem)
        video_prefix[(group_of(rel), stem)] = (ts.strftime("%Y%m%d"), ts.strftime("%H%M"))

    moves: list[tuple[Path, Path]] = []
    for p in sorted(files):
        rel = p.relative_to(VIDEOS)
        grp = group_of(rel)
        stem = STAMP_RE.sub("", p.stem)
        # `foo.health.json` pairs with `foo.mp4`
        base = stem.split(".")[0]
        key = video_prefix.get((grp, stem)) or video_prefix.get((grp, base))
        if key is None:
            if p.suffix.lower() in SIDECAR_EXT and not (VIDEOS / rel).with_suffix(".mp4").exists():
                ts = datetime.fromtimestamp(p.stat().st_mtime)
                key = (ts.strftime("%Y%m%d"), ts.strftime("%H%M"))
            else:
                continue
        day, hhmm = key
        new = VIDEOS / f"{day}_{grp}" / f"{hhmm}_{stem}{p.suffix}"
        if new != p:
            moves.append((p, new))
    return moves


def text_files() -> list[Path]:
    out: list[Path] = []
    for pattern in TEXT_GLOBS:
        for p in ROOT.rglob(pattern):
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            out.append(p)
    return out


def rewrite_refs(moves: list[tuple[Path, Path]], apply: bool) -> int:
    """Replace old video paths with new ones in tracked text files."""
    sub = {str(o.relative_to(ROOT)): str(n.relative_to(ROOT)) for o, n in moves}
    # also the docs-relative form ("videos/reorient/x.mp4")
    for o, n in moves:
        sub[str(o.relative_to(ROOT / "docs/rl"))] = str(n.relative_to(ROOT / "docs/rl"))
    ordered = sorted(sub.items(), key=lambda kv: -len(kv[0]))
    touched = 0
    for f in text_files():
        try:
            text = f.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        new = text
        for old_s, new_s in ordered:
            if old_s in new:
                new = new.replace(old_s, new_s)
        if new != text:
            touched += 1
            if apply:
                f.write_text(new)
    return touched


def write_index(apply: bool) -> None:
    if not apply:
        return
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(VIDEOS.rglob("*")):
        if p.is_file() and p.suffix.lower() in MEDIA_EXT:
            groups[p.parent.name].append(p)
    lines = ["# Video index",
             "",
             "Auto-generated by `scripts/reorganize_videos.py` — do not hand-edit.",
             "Folders are `<YYYYMMDD>_<experiment>`, files `<HHMM>_<name>`, so plain",
             "alphabetical order is chronological order.",
             ""]
    for g in sorted(groups, reverse=True):
        lines.append(f"## {g}  ({len(groups[g])} clips)")
        lines.append("")
        for p in sorted(groups[g]):
            lines.append(f"- [{p.name}]({p.relative_to(VIDEOS.parent)})")
        lines.append("")
    (VIDEOS / "INDEX.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="execute (default: dry run)")
    args = ap.parse_args()

    moves = plan()
    if not moves:
        print("[videos] nothing to move — already organized")
    for old, new in moves[:15]:
        print(f"  {old.relative_to(VIDEOS)}  ->  {new.relative_to(VIDEOS)}")
    if len(moves) > 15:
        print(f"  ... and {len(moves) - 15} more")
    print(f"[videos] {len(moves)} files to move")

    if args.apply:
        for old, new in moves:
            new.parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(["git", "mv", str(old), str(new)], cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:  # untracked file: plain move
                old.rename(new)
        # prune now-empty dirs
        for d in sorted(VIDEOS.rglob("*"), key=lambda p: -len(p.parts)):
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    touched = rewrite_refs(moves, args.apply)
    print(f"[videos] {'rewrote' if args.apply else 'would rewrite'} refs in {touched} text files")
    write_index(args.apply)
    if args.apply:
        print(f"[videos] wrote {VIDEOS / 'INDEX.md'}")


if __name__ == "__main__":
    main()
