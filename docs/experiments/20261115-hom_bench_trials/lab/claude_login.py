"""When the Claude login on this machine ends, for the runners that work unattended (2026-11-20: the login of the
session watching the bench reached its fixed time limit before 08:04, "this login reached its time limit and cannot be
renewed"; the morning series started at 09:17 and stopped at 09:21 with the sphere held 49 mm above the table, and
nothing could act on it until the owner logged in again at 10:33).

A claude.ai login lasts about four weeks from the sign-in, and using Claude Code does not extend it; `claude auth status
--text` gives its end as two dates, and the docs say to sign in again before the earlier one
(code.claude.com/docs/en/authentication, "Check when a stored login ends"). /login in any Claude Code window on the machine
renews it for every session there. The long-lived token of `claude setup-token` does not fit the session that watches
the bench: it "can only make model requests, so it can't establish Remote Control sessions or fetch claude.ai
connectors", and it outranks the browser login.

hom_series.py and wait_for_lights.py print line() when they start. It reads only the expiry field of
~/.claude/.credentials.json, never a token.

    /usr/bin/python3 claude_login.py
"""
import datetime as dt
import json
from pathlib import Path

CREDS = Path.home() / ".claude/.credentials.json"
MARGIN = dt.timedelta(days=3)          # the earlier end date of `claude auth status` (2026-12-16 for 2026-12-19 02:10 UTC)


def ends():
    """(renew by, ends) as local datetimes, or None without a claude.ai login on this machine."""
    try:
        ms = json.loads(CREDS.read_text())["claudeAiOauth"]["refreshTokenExpiresAt"]
    except (OSError, KeyError, TypeError, ValueError):
        return None
    end = dt.datetime.fromtimestamp(ms / 1000)
    return end - MARGIN, end


def line(warn_days=2.0):
    """One line for a runner's log; it starts with WARNING within warn_days of the renew-by date or past it."""
    e = ends()
    if e is None:
        return "Claude login: none on this machine"
    renew, end = e
    left = (renew - dt.datetime.now()).total_seconds() / 86400
    s = f"Claude login on this machine: renew by {renew:%Y-%m-%d %H:%M} ({left:+.1f} days), ends {end:%Y-%m-%d %H:%M}"
    if left < warn_days:
        s = ("WARNING: " + s + ". The session watching this run may lose its login before the run ends; /login in any "
             "Claude Code window here renews it")
    return s


if __name__ == "__main__":
    print(line())
