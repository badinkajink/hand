"""Fire the actual poke: run a configurable command (default ``claude -p``).

The command is intentionally generic. The default keeps a Claude usage window
warm with a trivial prompt, but ``command`` can be *any* argv -- a shell script,
a slash-command invocation, a background work-queue processor -- so the idle
quota can be spent on something useful.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import Config

_EXCERPT = 2000  # cap captured output so state/logs stay small


@dataclass
class TriggerResult:
    started_at: datetime
    finished_at: datetime
    returncode: int | None
    ok: bool
    stdout_excerpt: str
    stderr_excerpt: str
    dry_run: bool

    @property
    def duration_s(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


def build_command(config: Config) -> list[str]:
    """Materialize the argv, substituting ``{prompt}`` with ``config.prompt``.

    If no argument contains the ``{prompt}`` placeholder and the command still
    looks like the default ``claude -p`` form, the prompt is appended so the
    call is never sent empty.
    """
    argv = [part.replace("{prompt}", config.prompt) for part in config.command]
    if not any("{prompt}" in part for part in config.command):
        if config.prompt and config.command and config.command[0] == "claude" and "-p" in config.command:
            if config.prompt not in argv:
                argv.append(config.prompt)
    return argv


def describe_command(config: Config) -> str:
    return " ".join(shlex.quote(p) for p in build_command(config))


def run_trigger(config: Config) -> TriggerResult:
    """Execute the poke (or simulate it under ``dry_run``)."""
    argv = build_command(config)
    started = datetime.now(timezone.utc)

    if config.dry_run:
        return TriggerResult(
            started_at=started,
            finished_at=started,
            returncode=None,
            ok=True,
            stdout_excerpt="[dry-run] " + describe_command(config),
            stderr_excerpt="",
            dry_run=True,
        )

    try:
        proc = subprocess.run(
            argv,
            cwd=config.cwd,
            timeout=config.timeout_seconds,
            capture_output=True,
            text=True,
        )
        finished = datetime.now(timezone.utc)
        return TriggerResult(
            started_at=started,
            finished_at=finished,
            returncode=proc.returncode,
            ok=proc.returncode == 0,
            stdout_excerpt=(proc.stdout or "")[-_EXCERPT:],
            stderr_excerpt=(proc.stderr or "")[-_EXCERPT:],
            dry_run=False,
        )
    except FileNotFoundError as exc:
        finished = datetime.now(timezone.utc)
        return TriggerResult(started, finished, None, False, "", f"command not found: {exc}", False)
    except subprocess.TimeoutExpired as exc:
        finished = datetime.now(timezone.utc)
        return TriggerResult(started, finished, None, False, "", f"timed out after {exc.timeout}s", False)
