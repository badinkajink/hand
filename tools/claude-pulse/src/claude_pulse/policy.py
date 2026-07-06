"""The non-LLM decision engine: should we poke right now?

Everything here is a pure function of a :class:`Snapshot` + :class:`Config` +
``now``, so it is trivially unit-testable and has no side effects. The rules are
applied in order; the first veto wins and carries a human-readable reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

from .blocks import Block
from .config import Config


@dataclass
class Snapshot:
    """Everything the policy needs to decide, gathered once per tick."""

    now: datetime
    active_block: Block | None
    last_usage_time: datetime | None  # most recent real assistant message
    last_trigger_time: datetime | None  # most recent poke we fired
    pokes_this_window: int

    @property
    def last_activity(self) -> datetime | None:
        """Most recent activity of any kind (real usage OR our own poke)."""
        candidates = [t for t in (self.last_usage_time, self.last_trigger_time) if t]
        return max(candidates) if candidates else None


@dataclass
class Decision:
    should_trigger: bool
    reason: str


def _parse_hhmm(s: str) -> time | None:
    try:
        hh, mm = s.strip().split(":")
        return time(int(hh), int(mm))
    except (ValueError, AttributeError):
        return None


def in_quiet_hours(now_local: datetime, ranges: list[str]) -> bool:
    """True if ``now_local`` falls inside any ``"HH:MM-HH:MM"`` range.

    Ranges are interpreted in local time and may wrap past midnight
    (e.g. ``"23:00-07:00"``).
    """
    t = now_local.time()
    for r in ranges:
        try:
            start_s, end_s = r.split("-")
        except ValueError:
            continue
        start, end = _parse_hhmm(start_s), _parse_hhmm(end_s)
        if start is None or end is None:
            continue
        if start <= end:
            if start <= t < end:
                return True
        else:  # wraps midnight
            if t >= start or t < end:
                return True
    return False


def decide(config: Config, snap: Snapshot, *, now_local: datetime | None = None) -> Decision:
    """Return whether to fire a poke now, with a reason string either way."""
    now = snap.now
    now_local = now_local or now.astimezone()

    # 1. Quiet hours -----------------------------------------------------
    if in_quiet_hours(now_local, config.quiet_hours):
        return Decision(False, "in quiet hours")

    # 2. Window availability --------------------------------------------
    block = snap.active_block
    if block is None:
        if config.require_active_window:
            return Decision(
                False, "no active usage window and require_active_window is set"
            )
        # else: allowed to open a fresh window; fall through to idle check.
    else:
        remaining = block.time_remaining(now)
        min_left = timedelta(minutes=config.min_window_left_minutes)
        if remaining < min_left:
            mins = remaining.total_seconds() / 60
            return Decision(False, f"window resets in {mins:.0f}m (< {config.min_window_left_minutes:.0f}m)")

        # 3. Per-window token budget ------------------------------------
        if config.window_token_budget is not None:
            used = block.tokens.total
            if used >= config.window_token_budget:
                return Decision(
                    False, f"window budget reached ({used:,} >= {config.window_token_budget:,} tokens)"
                )

        # 4. Per-window poke cap ----------------------------------------
        if config.max_pokes_per_window is not None:
            if snap.pokes_this_window >= config.max_pokes_per_window:
                return Decision(
                    False, f"already poked {snap.pokes_this_window}x this window (max {config.max_pokes_per_window})"
                )

    # 5. Idle gate (the core rule) --------------------------------------
    idle_needed = timedelta(minutes=config.poke_after_idle_minutes)
    last = snap.last_activity
    if last is not None:
        idle = now - last
        if idle < idle_needed:
            mins = idle.total_seconds() / 60
            return Decision(
                False, f"only idle {mins:.0f}m (need {config.poke_after_idle_minutes:.0f}m)"
            )
        idle_mins = idle.total_seconds() / 60
        return Decision(True, f"idle {idle_mins:.0f}m >= {config.poke_after_idle_minutes:.0f}m")

    return Decision(True, "no prior activity recorded; eligible to poke")


def next_eligible_time(config: Config, snap: Snapshot) -> datetime | None:
    """Best estimate of when the idle gate next opens (for status display)."""
    last = snap.last_activity
    if last is None:
        return snap.now
    return last + timedelta(minutes=config.poke_after_idle_minutes)
