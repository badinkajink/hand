"""Persistent state: a record of the pokes *we* have fired.

Kept separate from the usage logs so the policy can reason about "when did I
last poke" without having to distinguish our pokes from the user's real work in
the JSONL. Writes are atomic (temp file + rename); a small advisory file lock
keeps a daemon and a stray cron ``tick`` from stepping on each other.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

_MAX_HISTORY = 500


@dataclass
class TriggerRecord:
    at: str  # ISO-8601 UTC
    ok: bool
    returncode: int | None
    duration_s: float
    dry_run: bool
    reason: str

    @property
    def when(self) -> datetime:
        return datetime.fromisoformat(self.at)


class State:
    """Small JSON-backed store of our trigger history."""

    def __init__(self, path: Path):
        self.path = path
        self.triggers: list[TriggerRecord] = []

    @classmethod
    def load(cls, path: Path) -> "State":
        self = cls(path)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, ValueError):
                data = {}
            for item in data.get("triggers", []):
                try:
                    self.triggers.append(TriggerRecord(**item))
                except TypeError:
                    continue
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"triggers": [asdict(t) for t in self.triggers[-_MAX_HISTORY:]]}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    def record(self, record: TriggerRecord) -> None:
        self.triggers.append(record)
        self.save()

    def last_trigger_time(self) -> datetime | None:
        if not self.triggers:
            return None
        return max(t.when for t in self.triggers)

    def triggers_since(self, since: datetime) -> list[TriggerRecord]:
        return [t for t in self.triggers if t.when >= since]


@contextmanager
def file_lock(path: Path, *, blocking: bool = False) -> Iterator[bool]:
    """Advisory exclusive lock via ``fcntl`` (best-effort; POSIX only).

    Yields ``True`` if the lock was acquired, ``False`` if it was already held
    (and ``blocking`` is False). On platforms without ``fcntl`` it degrades to a
    no-op that always yields ``True``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX
        yield True
        return

    fh = path.open("w")
    flags = fcntl.LOCK_EX
    if not blocking:
        flags |= fcntl.LOCK_NB
    try:
        fcntl.flock(fh.fileno(), flags)
    except OSError:
        fh.close()
        yield False
        return
    try:
        fh.write(f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}\n")
        fh.flush()
        yield True
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()
