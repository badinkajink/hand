"""Orchestration: build a snapshot, decide, maybe poke, record; loop for daemon."""

from __future__ import annotations

import logging
import signal
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone

from .blocks import Block, active_block, compute_blocks
from .config import Config
from .policy import Decision, Snapshot, decide, next_eligible_time
from .state import State, TriggerRecord, file_lock
from .trigger import TriggerResult, run_trigger
from .usage import load_usage_entries, recent_lookback

log = logging.getLogger("claude_pulse")


@dataclass
class TickResult:
    snapshot: Snapshot
    decision: Decision
    trigger: TriggerResult | None


def build_snapshot(config: Config, state: State, *, now: datetime | None = None) -> Snapshot:
    """Read recent usage, reconstruct windows, and assemble policy inputs."""
    now = now or datetime.now(timezone.utc)
    since = now - recent_lookback(config.session_hours)
    entries = load_usage_entries(config.resolved_data_dirs(), since=since)

    blocks = compute_blocks(entries, session_hours=config.session_hours)
    block = active_block(blocks, now)

    last_usage = entries[-1].timestamp if entries else None
    last_trigger = state.last_trigger_time()

    pokes_this_window = 0
    if block is not None:
        pokes_this_window = len(state.triggers_since(block.start))

    return Snapshot(
        now=now,
        active_block=block,
        last_usage_time=last_usage,
        last_trigger_time=last_trigger,
        pokes_this_window=pokes_this_window,
    )


def tick(config: Config, state: State, *, now: datetime | None = None) -> TickResult:
    """One evaluation cycle: decide, and poke + record if eligible."""
    snap = build_snapshot(config, state, now=now)
    decision = decide(config, snap)

    trigger: TriggerResult | None = None
    if decision.should_trigger:
        log.info("poking: %s", decision.reason)
        trigger = run_trigger(config)
        state.record(
            TriggerRecord(
                at=trigger.started_at.isoformat(),
                ok=trigger.ok,
                returncode=trigger.returncode,
                duration_s=round(trigger.duration_s, 3),
                dry_run=trigger.dry_run,
                reason=decision.reason,
            )
        )
        level = logging.INFO if trigger.ok else logging.WARNING
        log.log(
            level,
            "poke %s (rc=%s, %.1fs)%s",
            "ok" if trigger.ok else "FAILED",
            trigger.returncode,
            trigger.duration_s,
            "" if trigger.ok else f": {trigger.stderr_excerpt[:200]}",
        )
    else:
        log.debug("skip: %s", decision.reason)

    return TickResult(snapshot=snap, decision=decision, trigger=trigger)


class _Stopper:
    """Flip a flag on SIGINT/SIGTERM so the loop can exit cleanly."""

    def __init__(self) -> None:
        self.stop = False

    def __call__(self, *_a: object) -> None:  # signal handler signature
        self.stop = True


def run(config: Config) -> int:
    """Daemon loop: tick, sleep ``check_interval_minutes``, repeat.

    Guarded by an advisory lock so a second instance (or a stray cron ``tick``)
    won't run concurrently.
    """
    stopper = _Stopper()
    signal.signal(signal.SIGINT, stopper)
    signal.signal(signal.SIGTERM, stopper)

    interval_s = max(5.0, config.check_interval_minutes * 60.0)

    with file_lock(config.lock_path()) as acquired:
        if not acquired:
            log.error("another claude-pulse instance holds the lock; exiting")
            return 1
        log.info(
            "claude-pulse running: poke after %.0fm idle, check every %.0fm, window %.1fh%s",
            config.poke_after_idle_minutes,
            config.check_interval_minutes,
            config.session_hours,
            " [DRY-RUN]" if config.dry_run else "",
        )
        while not stopper.stop:
            try:
                state = State.load(config.state_path())
                tick(config, state)
            except Exception:  # keep the daemon alive across transient errors
                log.exception("tick failed; continuing")
            # Sleep in short slices so signals are handled promptly.
            slept = 0.0
            while slept < interval_s and not stopper.stop:
                _time.sleep(min(1.0, interval_s - slept))
                slept += 1.0

    log.info("claude-pulse stopped")
    return 0
