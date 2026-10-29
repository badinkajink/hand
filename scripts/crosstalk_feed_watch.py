#!/usr/bin/env python3
"""Watcher of the UR5 coordinator's crosstalk results for the doc-writing job (owner, 2026-10-27: "a periodic poller ... to
fetch results from the ur5 calibration coordinator and update results ... namely for the crosstalk docpage").

The coordinator appends one JSON line per update to docs/experiments/20261027-v3_tip_calibration_and_crosstalk/feed.jsonl
(`time`, `kind`, `summary`, `files`, `commit`; the README there, "Results feed for the doc-writing session") and commits
it; its result commits start with "Crosstalk:". This script only reads git. Every --interval seconds it reads the feed at
HEAD and the subjects of HEAD's last 300 commits. When the feed has lines past the count in --state, or a "Crosstalk:"
commit is not in --state, it waits until nothing more arrives for --settle seconds, prints the new lines and commits, and
exits 0. After --max-wait minutes without news it prints "no news" and exits 3. Commits are keyed by author date and
subject, which a rebase keeps.

    .venv/bin/python scripts/crosstalk_feed_watch.py [--once] [--ack] [--state logs/crosstalk_feed_watch/state.json]
        [--interval 120] [--settle 600] [--max-wait 230] [--until-gate]

--once compares once and exits (0 news, 3 none); --ack records what HEAD shows as handled. Without a state file every
line and commit counts as new. The autopilot job of docs/handoff/20261027-crosstalk_feed_poller.md runs it with
run_in_background; its exit wakes that session. The job runs it through `~/.claude/bin/resguard.sh run`, which refuses
to launch while the box's launch gate is closed and then also returns 3. --until-gate is for that case (10-28, gate
closed on swap): the watcher runs plain, at 19.5 MB, and exits 10 once `resguard.sh status` reports the gate open.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEED = "docs/experiments/20261027-v3_tip_calibration_and_crosstalk/feed.jsonl"
PREFIX = "Crosstalk:"


def git(*args):
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def show(path):
    """A file at HEAD; an LFS pointer is resolved from the local LFS store (the feed is an LFS file)."""
    blob = git("show", f"HEAD:{path}")
    if blob.startswith("version https://git-lfs.github.com/spec/v1"):
        try:
            r = subprocess.run(["git", "-C", str(ROOT), "lfs", "smudge", "--", path], input=blob, capture_output=True,
                               text=True, timeout=60)
            return r.stdout if r.returncode == 0 else ""
        except subprocess.TimeoutExpired:
            return ""
    return blob


def snapshot():
    """The feed's lines and the "Crosstalk:" commits at HEAD."""
    head = git("rev-parse", "--short=8", "HEAD").strip()
    lines = [ln for ln in show(FEED).splitlines() if ln.strip()]
    commits = []
    for row in git("log", "-300", "--format=%h%x09%aI%x09%s").splitlines():
        h, ad, subj = row.split("\t", 2)
        if subj.startswith(PREFIX):
            commits.append(dict(hash=h, key=f"{ad} {subj}", subject=subj, date=ad[5:16].replace("T", " ")))
    return dict(head=head, lines=lines, commits=commits)


def gate_open():
    """resguard's launch gate (`resguard.sh status` exits 0 when it is open)."""
    try:
        r = subprocess.run([str(Path.home() / ".claude/bin/resguard.sh"), "status"], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def sha(lines):
    return hashlib.sha1("\n".join(lines).encode()).hexdigest()


def load_state(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return dict(n_lines=0, prefix_sha=sha([]), commit_keys=[])


def news(snap, st):
    """New feed lines (and whether the handled ones changed) and unacknowledged commits."""
    n = st.get("n_lines", 0)
    rewritten = len(snap["lines"]) < n or sha(snap["lines"][:n]) != st.get("prefix_sha", sha([]))
    lines = snap["lines"] if rewritten else snap["lines"][n:]
    seen = set(st.get("commit_keys", []))
    commits = [c for c in snap["commits"] if c["key"] not in seen]
    return dict(lines=lines, rewritten=rewritten, commits=commits)


def report(snap, nw, first):
    print(f"HEAD {snap['head']}; feed {len(snap['lines'])} lines; news first seen {first}")
    if nw["rewritten"]:
        print("the feed's handled lines changed (rewritten or shortened): all lines follow")
    for ln in nw["lines"]:
        print(ln)
    if nw["commits"]:
        print(f'"{PREFIX}" commits not acknowledged:')
        for c in nw["commits"]:
            print(f"  {c['hash']} {c['date']} {c['subject']}")


def ack(path, snap):
    st = dict(n_lines=len(snap["lines"]), prefix_sha=sha(snap["lines"]), commit_keys=[c["key"] for c in snap["commits"]],
              head=snap["head"], acked=time.strftime("%Y-%m-%d %H:%M:%S"))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)
    print(f"acknowledged {st['n_lines']} feed lines and {len(st['commit_keys'])} commits at HEAD {snap['head']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", default=str(ROOT / "logs/crosstalk_feed_watch/state.json"))
    ap.add_argument("--interval", type=float, default=120.0, help="seconds between reads")
    ap.add_argument("--settle", type=float, default=600.0, help="seconds without further news before reporting")
    ap.add_argument("--max-wait", type=float, default=230.0, help="minutes before exiting with no news")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--ack", action="store_true")
    ap.add_argument("--until-gate", action="store_true", help="exit 10 once resguard's launch gate is open")
    a = ap.parse_args()
    if a.ack:
        ack(a.state, snapshot())
        return 0
    st = load_state(a.state)
    t0, first, last_sig = time.time(), None, None
    while True:
        snap = snapshot()
        nw = news(snap, st)
        has = bool(nw["lines"] or nw["commits"] or nw["rewritten"])
        if a.once:
            if has:
                report(snap, nw, time.strftime("%H:%M:%S"))
                return 0
            print(f"no news at HEAD {snap['head']} ({len(snap['lines'])} feed lines)")
            return 3
        if has:
            sig = (len(snap["lines"]), len(nw["commits"]), snap["head"])
            if sig != last_sig:
                first = first or time.strftime("%H:%M:%S")
                last_sig, t_sig = sig, time.time()
            elif time.time() - t_sig >= a.settle:
                report(snap, nw, first)
                return 0
        if not has and time.time() - t0 >= 60 * a.max_wait:
            print(f"no news at HEAD {snap['head']} after {a.max_wait:.0f} min ({len(snap['lines'])} feed lines); "
                  f"{time.strftime('%Y-%m-%d %H:%M')}")
            return 3
        if a.until_gate and not has and gate_open():
            print(f"no news at HEAD {snap['head']} ({len(snap['lines'])} feed lines); resguard's launch gate open at "
                  f"{time.strftime('%Y-%m-%d %H:%M')}")
            return 10
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
