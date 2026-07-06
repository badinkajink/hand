"""Reconstruct Claude's rolling ~5-hour usage windows ("blocks").

Anthropic subscription plans meter usage in rolling windows that reset a fixed
number of hours (5, by default) after the *first* activity in the window. There
is no local API for "tokens remaining", but we can reconstruct the windows from
the usage logs the same way ``ccusage`` does and report:

* when the current window started and when it resets,
* how many tokens have been spent in it,
* whether a window is currently active at all.

The reconstruction is a model, not ground truth (the real reset is tied to the
server-side first-message time), but it is accurate enough to schedule top-up
pokes and to show a useful countdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .usage import UsageEntry


@dataclass
class TokenCounts:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    messages: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_tokens
            + self.cache_read_tokens
        )

    def add(self, entry: UsageEntry) -> None:
        self.input_tokens += entry.input_tokens
        self.output_tokens += entry.output_tokens
        self.cache_creation_tokens += entry.cache_creation_tokens
        self.cache_read_tokens += entry.cache_read_tokens
        self.messages += 1


@dataclass
class Block:
    """One reconstructed usage window."""

    start: datetime  # floored to the hour, like ccusage
    duration: timedelta
    first_entry: datetime
    last_entry: datetime
    tokens: TokenCounts = field(default_factory=TokenCounts)

    @property
    def resets_at(self) -> datetime:
        """Nominal reset time: window start + duration."""
        return self.start + self.duration

    def is_active(self, now: datetime) -> bool:
        """True if ``now`` is inside the window and activity is recent.

        Mirrors ccusage: a block is active only while both the window itself
        and the time since its last entry are within ``duration``.
        """
        return (now - self.start) < self.duration and (now - self.last_entry) < self.duration

    def time_remaining(self, now: datetime) -> timedelta:
        """Time until this window resets (clamped at zero)."""
        remaining = self.resets_at - now
        return remaining if remaining > timedelta(0) else timedelta(0)


def _floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def compute_blocks(
    entries: list[UsageEntry],
    *,
    session_hours: float = 5.0,
) -> list[Block]:
    """Group time-sorted usage entries into rolling windows.

    A new window starts when either (a) more than ``session_hours`` have passed
    since the current window's start, or (b) more than ``session_hours`` of
    inactivity separate two consecutive entries. Window starts are floored to
    the hour, matching ccusage's convention.
    """
    duration = timedelta(hours=session_hours)
    ordered = sorted(entries, key=lambda e: e.timestamp)
    blocks: list[Block] = []

    current: Block | None = None
    for entry in ordered:
        ts = entry.timestamp
        if current is None:
            current = _new_block(entry, duration)
        else:
            since_start = ts - current.start
            since_last = ts - current.last_entry
            if since_start >= duration or since_last >= duration:
                blocks.append(current)
                current = _new_block(entry, duration)
            else:
                current.last_entry = ts
        current.tokens.add(entry)

    if current is not None:
        blocks.append(current)
    return blocks


def _new_block(entry: UsageEntry, duration: timedelta) -> Block:
    return Block(
        start=_floor_to_hour(entry.timestamp),
        duration=duration,
        first_entry=entry.timestamp,
        last_entry=entry.timestamp,
    )


def active_block(blocks: list[Block], now: datetime | None = None) -> Block | None:
    """Return the currently-active window, or ``None`` if idle past a reset."""
    now = now or datetime.now(timezone.utc)
    for block in reversed(blocks):
        if block.is_active(now):
            return block
    return None


def latest_block(blocks: list[Block]) -> Block | None:
    """Return the most recent window regardless of whether it is still active."""
    return blocks[-1] if blocks else None
