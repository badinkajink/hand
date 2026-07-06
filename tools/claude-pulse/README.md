# claude-pulse

A tiny, **non-LLM** monitor that watches your Claude usage windows (the ones
that "reset every 5 hours") and fires an intermittent **poke** to Claude when a
window has been sitting **idle** long enough — so paid quota that would
otherwise reset unused gets put to work.

- **Pure standard library. Zero dependencies.** One `pip install` (or just
  `python -m claude_pulse`) and it runs anywhere with Python 3.11+.
- **No LLM in the loop.** Decisions come from a small, deterministic,
  unit-tested policy engine reading Claude Code's own local logs.
- **Reusable.** Self-contained package — lift the whole `claude-pulse/`
  directory into its own repo and it Just Works.

> ⚠️ It only automates *your own account's* usage within *your own* limits.
> Keep the intervals sane (the defaults are), and use `quiet_hours` /
> `max_pokes_per_window` / `window_token_budget` if you want hard caps.

## How it works

1. **Read** — parses the JSONL transcripts Claude Code writes under
   `~/.claude/projects/**/*.jsonl`, extracting per-message token usage
   (de-duplicated by `message.id` + `requestId`, like [`ccusage`]).
2. **Reconstruct** — groups usage into rolling ~5-hour **windows** (a new
   window starts after 5h of elapsed time or 5h of inactivity), giving a
   reset countdown and per-window token totals.
3. **Decide** — a pure function says *poke* or *skip*. The core rule: only poke
   after `poke_after_idle_minutes` of inactivity, where **both your real work
   and our own pokes count as activity**. So it defers to you when you're
   working and self-spaces its pokes when you're not.
4. **Trigger** — runs a configurable command (default `claude -p <prompt>`).
   Point it at real background work to make the idle quota productive.

There is no public API for "tokens remaining", so the 5-hour windows are a
faithful *reconstruction* from the logs (same model `ccusage` uses), not a
server-truth number. That's plenty to schedule top-ups and show a countdown.

[`ccusage`]: https://github.com/ryoppippi/ccusage

## Install

```bash
# from this directory
pip install -e .          # or: uv pip install -e .
claude-pulse status

# or with no install at all:
python -m claude_pulse status     # (add src/ to PYTHONPATH if not installed)
```

## Usage

```bash
claude-pulse status          # current window, tokens, countdown, next poke
claude-pulse tick            # evaluate once; poke if eligible  (for cron/timers)
claude-pulse run             # foreground daemon loop
claude-pulse history         # pokes we have fired
claude-pulse config          # effective merged config + where it came from
claude-pulse deploy          # print cron/systemd setup snippets
```

Try it safely first with `--dry-run` (evaluates and logs, never runs the
command):

```bash
claude-pulse --dry-run --idle-minutes 120 tick -v
```

Example `status`:

```
claude-pulse 0.1.0   (2026-07-03 14:12:05 PDT)
  config source : ~/.config/claude-pulse/config.toml
  data dirs     : /home/me/.claude/projects

  usage window  : started 2026-07-03 12:00:00 PDT  (132m ago)
  resets in     : 2h48m  (at 2026-07-03 17:00:00 PDT)
  tokens used   : 1,204,882  (in 4,201 / out 88,004 / cache-w 402,110 / cache-r 710,567)

  last activity : 2026-07-03 12:40:11 PDT  (1h32m ago)
  last poke     : never
  poke command  : claude -p 'This is an automated keep-alive ping ...'

  decision now  : would skip  --  only idle 92m (need 120m)
  next eligible : 2026-07-03 14:40:11 PDT  (in 28m)
```

## Configuration

Defaults < TOML file < `CLAUDE_PULSE_*` env vars < CLI flags. Copy
[`claude-pulse.example.toml`](claude-pulse.example.toml) to
`~/.config/claude-pulse/config.toml`.

| Key | Default | Meaning |
|---|---|---|
| `poke_after_idle_minutes` | `120` | Poke only after this much inactivity. |
| `session_hours` | `5` | Rolling window length. |
| `check_interval_minutes` | `10` | Daemon re-evaluation cadence. |
| `min_window_left_minutes` | `20` | Don't poke if the window resets very soon. |
| `require_active_window` | `false` | If true, only top up windows already opened by real work (never open a fresh one). |
| `window_token_budget` | `none` | Stop poking once a window hits N tokens. |
| `max_pokes_per_window` | `none` | Cap our own pokes per window. |
| `quiet_hours` | `[]` | Local-time ranges to never poke, e.g. `["01:00-06:00"]`. |
| `command` | `["claude","-p","{prompt}"]` | The poke. `{prompt}` is substituted. |
| `prompt` | keep-alive text | Prompt for the default command. |
| `cwd`, `timeout_seconds`, `dry_run`, `state_dir`, `log_file` | — | See example config. |

### Make the poke useful

The trigger is just *"run this command when there's idle quota"*. Instead of a
no-op keep-alive, point it at real work:

```toml
command = ["claude", "-p", "{prompt}"]
prompt  = "Read TODO.md, implement the top unchecked item, run the tests, and commit."
cwd     = "/home/me/project"
```

or run an arbitrary script / agent:

```toml
command = ["bash", "/home/me/bin/nightly_agent.sh"]
```

## Running it continuously

`claude-pulse deploy` prints ready-to-paste snippets. Templates live in
[`deploy/`](deploy):

- **cron** — [`deploy/crontab.example`](deploy/crontab.example)
- **systemd daemon** — [`deploy/claude-pulse.service`](deploy/claude-pulse.service)
- **systemd timer** (cron-free) — [`deploy/claude-pulse.timer`](deploy/claude-pulse.timer)
  + [`claude-pulse-tick.service`](deploy/claude-pulse-tick.service)
- **macOS launchd** — [`deploy/com.claude-pulse.plist`](deploy/com.claude-pulse.plist)

A frequent tick is fine: the policy engine gates the actual poke, so ticking
every 10 minutes just means "poke promptly once idle long enough".

## Tests

```bash
python -m unittest discover -s tests -v     # zero deps
```

## Design notes

- **Idle-based, not clock-based.** "Every 2 hours" is expressed as *2 hours of
  inactivity*, which is what "don't waste idle quota" actually means — it never
  fights your real usage.
- **Concurrency-safe.** The daemon holds an advisory lock; state writes are
  atomic (temp + rename). A stray cron `tick` next to a running daemon won't
  double-poke.
- **Portable.** No third-party packages, all times tz-aware UTC internally,
  local time only for display and `quiet_hours`.

## License

MIT.
