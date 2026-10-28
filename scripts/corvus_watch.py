#!/usr/bin/env python3
"""Wait until the UR5 lab computer corvus changes, then print what changed and exit (owner, 2026-10-27: renders and plans
that follow the night's registrations without anyone at the workstation). Read-only: every poll logs in as will over
Tailscale with the password in the repository's gitignored .dotenv (~/.local/bin/corvus-askpass), as
scripts/corvus_pull.sh does, and reads the newest session folder under ~/workspace/hand/logs/*-tip_calibration.

It exits on: a new session folder; a change of stage in the newest session (an action starts in another stage than the
one before it, as the hovers after the touch-offs, whose onsets are then in touches.json); a session's end (the
program's closing "ft_reads" event); an action that ends in "abort" or "error"; a new commit on origin/main; or, as a
heartbeat, --max seconds without any of these. Run it in the background and act on its last line.

    .venv/bin/python scripts/corvus_watch.py [--poll 120] [--max 3600]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOST = "will@100.93.255.76"
REMOTE = r'''
import glob, json, os, time
logs = os.path.expanduser("~/workspace/hand/logs")
ds = sorted(glob.glob(logs + "/2026*-tip_calibration/*/"))
s = ds[-1] if ds else ""
out = dict(session=s[len(logs) + 1:], now=time.time())
try:
    out["touches"] = len(json.load(open(s + "touches.json")))
except Exception:
    out["touches"] = None
try:
    ev = open(s + "events.jsonl").read().splitlines()
    out["events"] = len(ev)
    out["tail"] = [e[:400] for e in ev[-20:]]
    out["mtime"] = os.path.getmtime(s + "events.jsonl")
except Exception:
    out["events"], out["tail"] = 0, []
print(json.dumps(out))
'''


def probe():
    env = dict(os.environ, SSH_ASKPASS=str(Path.home() / ".local/bin/corvus-askpass"), SSH_ASKPASS_REQUIRE="force")
    try:
        r = subprocess.run(["ssh", "-o", "PubkeyAuthentication=no", "-o",
                            "PreferredAuthentications=password,keyboard-interactive", "-o", "ConnectTimeout=10", HOST,
                            "python3", "-"], input=REMOTE, env=env, capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def origin_head():
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "ls-remote", "origin", "refs/heads/main"], capture_output=True,
                           text=True, timeout=60)
        return r.stdout.split()[0] if r.stdout else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--poll", type=float, default=120.0)
    ap.add_argument("--max", type=float, default=3600.0)
    a = ap.parse_args()
    t0 = time.time()
    s0, h0 = None, origin_head()
    while s0 is None and time.time() - t0 < 600:
        s0 = probe()
        if s0 is None:
            time.sleep(30)
    if s0 is None:
        print("corvus unreachable for 10 min")
        return 2
    print(f"{time.strftime('%H:%M')} watching {s0['session']}: {s0['touches']} touches, {s0['events']} events", flush=True)
    fails = 0
    stage = None
    for e in s0["tail"]:
        try:
            ev = json.loads(e)
        except ValueError:
            continue
        if ev.get("event") == "start" and ev.get("stage"):
            stage = ev["stage"]
    while True:
        time.sleep(a.poll)
        s = probe()
        stamp = time.strftime("%H:%M")
        if s is None:
            fails += 1
            if fails * a.poll >= 900:
                print(f"{stamp} corvus unreachable for {fails} polls")
                return 2
            continue
        fails = 0
        k = min(20, s["events"] - s0["events"])
        new = s["tail"][-k:] if s["session"] == s0["session"] and k > 0 else []
        if s["session"] != s0["session"]:
            print(f"{stamp} new session {s['session']} ({s['events']} events): {s['tail'][-1:]}")
            return 0
        for e in new:
            if '"ft_reads"' in e:
                print(f"{stamp} session {s['session']} ended ({s['touches']} touches): {s['tail'][-2:]}")
                return 0
            if '"status": "abort"' in e or '"status": "error"' in e:
                print(f"{stamp} {s['session']}: {e}")
                return 0
            try:
                ev = json.loads(e)
            except ValueError:
                continue
            if ev.get("event") == "start" and ev.get("stage"):
                if stage is not None and ev["stage"] != stage:
                    print(f"{stamp} {s['session']}: stage {stage} -> {ev['stage']} at {ev.get('id')} "
                          f"({s['touches']} touches)")
                    return 0
                stage = ev["stage"]
        h = origin_head()
        if h and h0 and h != h0:
            print(f"{stamp} new commit on origin/main {h[:8]} ({s['session']}: {s['events']} events)")
            return 0
        s0 = s
        if time.time() - t0 > a.max:
            print(f"{stamp} heartbeat: {s['session']}, {s['touches']} touches, {s['events']} events, last "
                  f"{s['tail'][-1][:200] if s['tail'] else ''}")
            return 0


if __name__ == "__main__":
    sys.exit(main())
