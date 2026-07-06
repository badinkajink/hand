"""Read and parse Claude Code's local usage logs.

Claude Code (the CLI/IDE) writes one JSONL transcript per session under
``~/.claude/projects/<encoded-project>/<session-id>.jsonl``. Assistant
messages carry a ``message.usage`` block with token counts. We parse those
into :class:`UsageEntry` records, de-duplicated the same way ``ccusage`` does
(by ``message.id`` + ``requestId``), so double-counting across resumed
sessions or overlapping files does not happen.

Nothing here talks to the network or an LLM -- it is pure log reading.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class UsageEntry:
    """One assistant message's token usage, extracted from a JSONL line."""

    timestamp: datetime  # tz-aware, UTC
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    model: str | None
    message_id: str | None
    request_id: str | None
    session_id: str | None
    project: str | None
    source_file: str

    @property
    def total_tokens(self) -> int:
        """All token classes summed (input + output + both cache classes)."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
        )

    @property
    def dedup_key(self) -> str | None:
        """Stable identity for de-duplication, or ``None`` if unidentifiable."""
        if self.message_id and self.request_id:
            return f"{self.message_id}:{self.request_id}"
        return self.message_id or self.request_id


def default_data_dirs() -> list[Path]:
    """Locations Claude Code may keep its per-project JSONL logs.

    Honors ``CLAUDE_CONFIG_DIR`` (Claude Code's own override) and the common
    ``~/.claude`` / ``~/.config/claude`` roots. Only existing dirs returned.
    """
    candidates: list[Path] = []
    env_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser() / "projects")
    candidates.append(Path("~/.claude/projects").expanduser())
    candidates.append(Path("~/.config/claude/projects").expanduser())

    seen: set[Path] = set()
    out: list[Path] = []
    for c in candidates:
        rc = c.resolve()
        if rc in seen:
            continue
        seen.add(rc)
        if c.is_dir():
            out.append(c)
    return out


def _parse_timestamp(raw: str) -> datetime | None:
    """Parse an ISO-8601 timestamp (``...Z`` allowed) into a UTC datetime."""
    if not raw:
        return None
    try:
        # Python 3.11+ fromisoformat accepts the trailing 'Z'.
        dt = datetime.fromisoformat(raw)
    except ValueError:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _entry_from_record(record: dict, source_file: str) -> UsageEntry | None:
    """Build a :class:`UsageEntry` from one parsed JSONL object, or ``None``.

    Returns ``None`` for lines without an assistant ``message.usage`` block
    (user turns, tool results, summaries, meta lines, ...).
    """
    message = record.get("message")
    if not isinstance(message, dict):
        return None
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None

    ts = _parse_timestamp(record.get("timestamp", ""))
    if ts is None:
        return None

    def _int(container: dict, key: str) -> int:
        val = container.get(key, 0)
        return int(val) if isinstance(val, (int, float)) else 0

    return UsageEntry(
        timestamp=ts,
        input_tokens=_int(usage, "input_tokens"),
        output_tokens=_int(usage, "output_tokens"),
        cache_creation_tokens=_int(usage, "cache_creation_input_tokens"),
        cache_read_tokens=_int(usage, "cache_read_input_tokens"),
        model=message.get("model"),
        message_id=message.get("id"),
        request_id=record.get("requestId"),
        session_id=record.get("sessionId"),
        project=record.get("cwd") or _project_from_path(source_file),
        source_file=source_file,
    )


def _project_from_path(source_file: str) -> str | None:
    """Recover the project name from the encoded parent directory name."""
    parent = Path(source_file).parent.name
    return parent or None


def _iter_log_files(data_dirs: Iterable[Path], modified_since: datetime | None) -> Iterator[Path]:
    """Yield ``*.jsonl`` files, skipping ones untouched before ``modified_since``.

    Skipping by mtime is a safe optimization: Claude Code only *appends* to a
    transcript, so a file last written before the lookback window cannot
    contain entries inside it.
    """
    for d in data_dirs:
        for path in sorted(d.rglob("*.jsonl")):
            if modified_since is not None:
                try:
                    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                except OSError:
                    continue
                if mtime < modified_since:
                    continue
            yield path


def iter_usage_entries(
    data_dirs: Iterable[Path] | None = None,
    *,
    since: datetime | None = None,
) -> Iterator[UsageEntry]:
    """Yield de-duplicated :class:`UsageEntry` records from the logs.

    Parameters
    ----------
    data_dirs:
        Directories to scan. Defaults to :func:`default_data_dirs`.
    since:
        If given, only entries at/after this time are yielded, and files not
        modified since then are skipped entirely for speed.
    """
    dirs = list(data_dirs) if data_dirs is not None else default_data_dirs()
    seen: set[str] = set()

    for path in _iter_log_files(dirs, since):
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue  # tolerate partial/interleaved writes
                    if not isinstance(record, dict):
                        continue
                    entry = _entry_from_record(record, str(path))
                    if entry is None:
                        continue
                    if since is not None and entry.timestamp < since:
                        continue
                    key = entry.dedup_key
                    if key is not None:
                        if key in seen:
                            continue
                        seen.add(key)
                    yield entry
        except OSError:
            continue


def load_usage_entries(
    data_dirs: Iterable[Path] | None = None,
    *,
    since: datetime | None = None,
) -> list[UsageEntry]:
    """Eagerly load and time-sort usage entries (thin wrapper for callers)."""
    entries = list(iter_usage_entries(data_dirs, since=since))
    entries.sort(key=lambda e: e.timestamp)
    return entries


def recent_lookback(session_hours: float, factor: float = 3.0, margin_hours: float = 1.0) -> timedelta:
    """A lookback window guaranteed to contain the current 5h block + its gap.

    Blocks are at most ``session_hours`` long, so loading a few multiples of
    that (plus margin) is always enough to locate the *current* window's start
    without scanning the entire history.
    """
    return timedelta(hours=session_hours * factor + margin_hours)
