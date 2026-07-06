"""Configuration: dataclass + layered loader (defaults < file < env < CLI).

Kept deliberately plain (stdlib only) so the package drops into any project.
TOML config is optional; every field also has an env var and a CLI override.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    tomllib = None  # type: ignore[assignment]


ENV_PREFIX = "CLAUDE_PULSE_"

# The default poke: a trivial, tool-free prompt that costs almost nothing.
# Point ``command``/``prompt`` at real work to make the idle quota productive.
DEFAULT_PROMPT = (
    "This is an automated keep-alive ping from claude-pulse. "
    "Reply with exactly: pulse ok. Do not use any tools."
)
DEFAULT_COMMAND = ["claude", "-p", "{prompt}"]


@dataclass
class Config:
    """All knobs for the monitor. Times are minutes unless noted."""

    # --- where usage logs live -------------------------------------------
    data_dirs: list[str] = field(default_factory=list)  # empty => auto-detect
    session_hours: float = 5.0  # rolling window length ("resets every 5h")

    # --- policy: when to poke --------------------------------------------
    # Poke only after this much inactivity. Real usage AND our own pokes both
    # count as activity, so the monitor naturally defers to your real work and
    # spaces its own pokes by this interval.
    poke_after_idle_minutes: float = 120.0
    check_interval_minutes: float = 10.0  # daemon poll cadence
    # Skip if the window resets within this many minutes (a poke would open a
    # fresh window for almost no benefit).
    min_window_left_minutes: float = 20.0
    # If False, the monitor will open a brand-new window when none is active
    # (keeps availability "warm" overnight). If True, it only tops up windows
    # you already opened by real work.
    require_active_window: bool = False
    # Optional guard rails.
    window_token_budget: int | None = None  # skip once a window hits this many tokens
    max_pokes_per_window: int | None = None  # cap our own pokes per window
    quiet_hours: list[str] = field(default_factory=list)  # e.g. ["23:00-07:00"] local

    # --- the trigger ------------------------------------------------------
    command: list[str] = field(default_factory=lambda: list(DEFAULT_COMMAND))
    prompt: str = DEFAULT_PROMPT
    cwd: str | None = None
    timeout_seconds: float = 300.0
    dry_run: bool = False

    # --- bookkeeping ------------------------------------------------------
    state_dir: str = "~/.claude-pulse"
    log_file: str | None = None

    # -- derived helpers ---------------------------------------------------
    def resolved_data_dirs(self) -> list[Path]:
        from .usage import default_data_dirs

        if self.data_dirs:
            return [Path(d).expanduser() for d in self.data_dirs]
        return default_data_dirs()

    def state_path(self) -> Path:
        return Path(self.state_dir).expanduser() / "state.json"

    def lock_path(self) -> Path:
        return Path(self.state_dir).expanduser() / "claude-pulse.lock"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

_LIST_FIELDS = {"data_dirs", "command", "quiet_hours"}
_INT_OR_NONE_FIELDS = {"window_token_budget", "max_pokes_per_window"}
_BOOL_FIELDS = {"require_active_window", "dry_run"}
_FLOAT_FIELDS = {
    "session_hours",
    "poke_after_idle_minutes",
    "check_interval_minutes",
    "min_window_left_minutes",
    "timeout_seconds",
}


def default_config_paths() -> list[Path]:
    """Config file search order (first existing wins)."""
    paths = []
    env_path = os.environ.get(f"{ENV_PREFIX}CONFIG")
    if env_path:
        paths.append(Path(env_path).expanduser())
    paths.append(Path("claude-pulse.toml"))  # project-local
    paths.append(Path("~/.config/claude-pulse/config.toml").expanduser())
    return paths


def find_config_file(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_file() else None
    for p in default_config_paths():
        if p.is_file():
            return p
    return None


def _coerce(name: str, value: Any) -> Any:
    """Coerce a raw (env-string or TOML) value to the field's type."""
    if name in _LIST_FIELDS:
        if isinstance(value, str):
            # comma-separated for env vars
            return [v.strip() for v in value.split(",") if v.strip()]
        return list(value)
    if name in _BOOL_FIELDS:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if name in _INT_OR_NONE_FIELDS:
        if value in (None, "", "none", "null"):
            return None
        return int(value)
    if name in _FLOAT_FIELDS:
        return float(value)
    return value


def _from_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    valid = {f.name for f in fields(Config)}
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        if key in valid:
            out[key] = _coerce(key, value)
    return out


def load_toml(path: Path) -> dict[str, Any]:
    if tomllib is None:  # pragma: no cover
        raise RuntimeError("tomllib unavailable (need Python 3.11+) to read TOML config")
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    # Accept both a flat table and a [claude-pulse]/[tool.claude-pulse] section.
    for section in ("claude-pulse", "claude_pulse"):
        if isinstance(data.get(section), dict):
            data = {**{k: v for k, v in data.items() if not isinstance(v, dict)}, **data[section]}
            break
    tool = data.get("tool")
    if isinstance(tool, dict) and isinstance(tool.get("claude-pulse"), dict):
        data = {**data, **tool["claude-pulse"]}
    return _from_mapping(data)


def load_env(environ: dict[str, str] | None = None) -> dict[str, Any]:
    environ = environ if environ is not None else dict(os.environ)
    valid = {f.name for f in fields(Config)}
    out: dict[str, Any] = {}
    for key, value in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        field_name = key[len(ENV_PREFIX):].lower()
        if field_name == "config":
            continue  # handled separately as the config path
        if field_name in valid:
            out[field_name] = _coerce(field_name, value)
    return out


def load_config(
    config_path: str | None = None,
    *,
    cli_overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> tuple[Config, Path | None]:
    """Merge defaults < TOML file < env < CLI. Returns (config, source_path)."""
    merged: dict[str, Any] = {}

    source = find_config_file(config_path)
    if source is not None:
        merged.update(load_toml(source))

    merged.update(load_env(environ))

    if cli_overrides:
        merged.update({k: v for k, v in cli_overrides.items() if v is not None})

    return Config(**merged), source
