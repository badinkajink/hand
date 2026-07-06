"""Command-line interface for claude-pulse.

Subcommands
-----------
status   show the current usage window, tokens, countdown, and next poke time
tick     evaluate once and poke if eligible (for cron / systemd timers)
run      run the monitor as a foreground daemon loop
history  list the pokes we have fired
config   print the effective merged configuration and its source
deploy   print ready-to-use cron / systemd snippets with resolved paths
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timezone

from . import __version__
from .config import Config, load_config
from .monitor import build_snapshot, run, tick
from .policy import decide, next_eligible_time
from .state import State
from .trigger import describe_command


# --------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------

def _fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return "never"
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _fmt_delta(seconds: float) -> str:
    seconds = int(seconds)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{sign}{h}h{m:02d}m"
    if m:
        return f"{sign}{m}m{s:02d}s"
    return f"{sign}{s}s"


def _fmt_tokens(n: int) -> str:
    return f"{n:,}"


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------

def cmd_status(config: Config, args: argparse.Namespace, source) -> int:
    now = datetime.now(timezone.utc)
    state = State.load(config.state_path())
    snap = build_snapshot(config, state, now=now)
    decision = decide(config, snap)
    nxt = next_eligible_time(config, snap)

    block = snap.active_block
    if args.json:
        payload = {
            "now": now.isoformat(),
            "config_source": str(source) if source else None,
            "active_window": None,
            "last_usage": snap.last_usage_time.isoformat() if snap.last_usage_time else None,
            "last_poke": snap.last_trigger_time.isoformat() if snap.last_trigger_time else None,
            "pokes_this_window": snap.pokes_this_window,
            "decision": {"should_trigger": decision.should_trigger, "reason": decision.reason},
            "next_eligible": nxt.isoformat() if nxt else None,
        }
        if block is not None:
            payload["active_window"] = {
                "start": block.start.isoformat(),
                "resets_at": block.resets_at.isoformat(),
                "time_remaining_s": block.time_remaining(now).total_seconds(),
                "tokens": asdict(block.tokens),
            }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"claude-pulse {__version__}   ({_fmt_dt(now)})")
    print(f"  config source : {source if source else '(defaults only)'}")
    print(f"  data dirs     : {', '.join(str(d) for d in config.resolved_data_dirs()) or '(none found)'}")
    print()
    if block is None:
        print("  usage window  : none active (idle past the last reset)")
    else:
        rem = block.time_remaining(now)
        used_min = (now - block.start).total_seconds() / 60
        print(f"  usage window  : started {_fmt_dt(block.start)}  ({used_min:.0f}m ago)")
        print(f"  resets in     : {_fmt_delta(rem.total_seconds())}  (at {_fmt_dt(block.resets_at)})")
        t = block.tokens
        print(f"  tokens used   : {_fmt_tokens(t.total)}  "
              f"(in {t.input_tokens:,} / out {t.output_tokens:,} / "
              f"cache-w {t.cache_creation_tokens:,} / cache-r {t.cache_read_tokens:,})")
        if config.window_token_budget:
            pct = 100.0 * t.total / config.window_token_budget
            print(f"  window budget : {_fmt_tokens(config.window_token_budget)}  ({pct:.0f}% used)")
        if config.max_pokes_per_window:
            print(f"  pokes/window  : {snap.pokes_this_window} / {config.max_pokes_per_window}")
    print()
    last_act = snap.last_activity
    print(f"  last activity : {_fmt_dt(last_act)}"
          + (f"  ({_fmt_delta((now - last_act).total_seconds())} ago)" if last_act else ""))
    print(f"  last poke     : {_fmt_dt(snap.last_trigger_time)}")
    print(f"  poke command  : {describe_command(config)}")
    print()
    verdict = "WOULD POKE" if decision.should_trigger else "would skip"
    print(f"  decision now  : {verdict}  --  {decision.reason}")
    if not decision.should_trigger and nxt is not None and nxt > now:
        print(f"  next eligible : {_fmt_dt(nxt)}  (in {_fmt_delta((nxt - now).total_seconds())})")
    return 0


def cmd_tick(config: Config, args: argparse.Namespace, source) -> int:
    state = State.load(config.state_path())
    result = tick(config, state)
    d = result.decision
    if result.trigger is not None:
        tr = result.trigger
        status = "dry-run" if tr.dry_run else ("ok" if tr.ok else "FAILED")
        print(f"poked ({status}, rc={tr.returncode}, {tr.duration_s:.1f}s): {d.reason}")
        return 0 if tr.ok else 2
    print(f"skipped: {d.reason}")
    return 0


def cmd_run(config: Config, args: argparse.Namespace, source) -> int:
    return run(config)


def cmd_history(config: Config, args: argparse.Namespace, source) -> int:
    state = State.load(config.state_path())
    records = state.triggers[-args.limit:]
    if not records:
        print("no pokes recorded yet")
        return 0
    if args.json:
        print(json.dumps([asdict(r) for r in records], indent=2))
        return 0
    for r in records:
        status = "dry" if r.dry_run else ("ok " if r.ok else "ERR")
        print(f"{_fmt_dt(r.when)}  [{status}] rc={r.returncode} {r.duration_s:>6.1f}s  {r.reason}")
    return 0


def cmd_config(config: Config, args: argparse.Namespace, source) -> int:
    if args.json:
        print(json.dumps(asdict(config), indent=2))
        return 0
    print(f"# effective configuration (source: {source if source else 'defaults only'})")
    for key, value in asdict(config).items():
        print(f"{key} = {value!r}")
    return 0


def cmd_deploy(config: Config, args: argparse.Namespace, source) -> int:
    exe = sys.argv[0] if sys.argv and sys.argv[0] else "claude-pulse"
    py = sys.executable
    interval = int(config.check_interval_minutes)
    print("# --- cron (one tick per check interval) --------------------------------")
    print(f"# run 'crontab -e' and add (every {interval}m):")
    print(f"*/{max(1, interval)} * * * * {py} -m claude_pulse tick >> ~/.claude-pulse/cron.log 2>&1")
    print()
    print("# --- systemd user service (daemon) ------------------------------------")
    print("# save to ~/.config/systemd/user/claude-pulse.service, then:")
    print("#   systemctl --user daemon-reload && systemctl --user enable --now claude-pulse")
    print("[Unit]")
    print("Description=claude-pulse idle-usage monitor")
    print("[Service]")
    print(f"ExecStart={py} -m claude_pulse run")
    print("Restart=on-failure")
    print("[Install]")
    print("WantedBy=default.target")
    return 0


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

def _add_global_args(p: argparse.ArgumentParser) -> None:
    """Global flags shared by the top parser and every subparser.

    Adding them in both places (via ``parents=``) lets the user write them on
    either side of the subcommand: ``claude-pulse -v tick`` or ``... tick -v``.
    """
    p.add_argument("-c", "--config", help="path to a TOML config file")
    p.add_argument("-v", "--verbose", action="count", default=0, help="-v info, -vv debug")
    p.add_argument("--data-dir", action="append", dest="data_dirs",
                   help="usage-log dir to scan (repeatable; overrides auto-detect)")
    p.add_argument("--session-hours", type=float, help="rolling window length (default 5)")
    p.add_argument("--idle-minutes", type=float, dest="poke_after_idle_minutes",
                   help="poke only after this many minutes idle (default 120)")
    p.add_argument("--check-minutes", type=float, dest="check_interval_minutes",
                   help="daemon poll cadence in minutes (default 10)")
    p.add_argument("--budget", type=int, dest="window_token_budget",
                   help="skip once the window reaches this many tokens")
    p.add_argument("--max-pokes", type=int, dest="max_pokes_per_window",
                   help="cap our own pokes per window")
    p.add_argument("--require-active-window", dest="require_active_window",
                   action="store_true", default=None,
                   help="only top up windows already opened by real work")
    p.add_argument("--prompt", help="prompt for the default 'claude -p' poke")
    p.add_argument("--command", help="override the poke command (shell-quoted; use {prompt})")
    p.add_argument("--cwd", help="working directory for the poke command")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", default=None,
                   help="evaluate and log, but never actually run the command")


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    _add_global_args(common)

    p = argparse.ArgumentParser(
        prog="claude-pulse",
        description="Non-LLM monitor that tops up idle Claude usage windows with scheduled pokes.",
        parents=[common],
    )
    p.add_argument("--version", action="version", version=f"claude-pulse {__version__}")

    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("status", parents=[common], help="show current window, tokens, and next poke")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("tick", parents=[common], help="evaluate once; poke if eligible (for cron)")
    s.set_defaults(func=cmd_tick)

    s = sub.add_parser("run", parents=[common], help="run the monitor as a daemon loop")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("history", parents=[common], help="list pokes we have fired")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_history)

    s = sub.add_parser("config", parents=[common], help="print the effective merged config")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_config)

    s = sub.add_parser("deploy", parents=[common], help="print cron/systemd setup snippets")
    s.set_defaults(func=cmd_deploy)

    return p


def _cli_overrides(args: argparse.Namespace) -> dict:
    import shlex

    keys = [
        "data_dirs", "session_hours", "poke_after_idle_minutes",
        "check_interval_minutes", "window_token_budget", "max_pokes_per_window",
        "require_active_window", "prompt", "cwd", "dry_run",
    ]
    overrides = {k: getattr(args, k, None) for k in keys}
    if getattr(args, "command", None):
        overrides["command"] = shlex.split(args.command)
    return {k: v for k, v in overrides.items() if v is not None}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")

    config, source = load_config(args.config, cli_overrides=_cli_overrides(args))

    if config.log_file:
        from pathlib import Path
        fh = logging.FileHandler(Path(config.log_file).expanduser())
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger("claude_pulse").addHandler(fh)

    func = getattr(args, "func", None)
    if func is None:
        # Default to status when no subcommand is given.
        return cmd_status(config, argparse.Namespace(json=False), source)
    return func(config, args, source)


if __name__ == "__main__":
    raise SystemExit(main())
