#!/usr/bin/env python3
"""Watcher of the UR5 coordinator's crosstalk results for the doc-writing job (owner, 2026-10-27: "a periodic poller ... to
fetch results from the ur5 calibration coordinator and update results ... namely for the crosstalk docpage").

The coordinator appends one JSON line per update to docs/experiments/20261027-v3_tip_calibration_and_crosstalk/feed.jsonl
(`time`, `kind`, `summary`, `files`, `commit`; the README there, "Results feed for the doc-writing session") and commits
it; its result commits start with "Crosstalk:". It also commits analyses with no feed line (owner, 2026-10-29 15:2x, when
the job widened to the calibration of every v3 tip). This script only reads git. Every --interval seconds it reads the
feed at HEAD and HEAD's last 300 commits with the files each changes. News is a feed line past the count in --state, or a
commit not in --state that starts with "Crosstalk:" or changes the v3 folder's data/ or README.md and is not this job's
own (subjects in MINE). Beside git it lists the coordinator's analysed holds of the v3 tips from ID 3 on
(logs/*v3idN*/holds.npz, N >= 3; logs/ is not in git), and a holds file not in --state is news too (step 38 of the note,
10-30). When news appears it waits until nothing more arrives for --settle seconds, prints the new lines, commits and
holds, and exits 0. After --max-wait minutes without news it prints "no news" and exits 3. Commits are keyed by
author date and subject, which a rebase keeps.

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
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V3 = "docs/experiments/20261027-v3_tip_calibration_and_crosstalk/"
FEED = V3 + "feed.jsonl"
PREFIX = "Crosstalk:"
# the coordinator also commits analyses without a feed line (owner, 10-29 15:2x): commits that add or change these, unless
# they are this job's own
WATCHED = (V3 + "data/", V3 + "README.md")
# the coordinator's analysis folders of a tip's sessions, logs/YYYYMMDD-{sphere,tall,rod}v3idN*/ with holds.npz,
# calib_rest.json and summary.json; tips from v3 ID 3 on
HOLDS_RE = re.compile(r"v3id(\d+)")
MINE = ("Fingertip crosstalk page:", "Calibration page:", "Fingertip overview:", "Physical-model page:",
        "Crosstalk feed poller note:", "Crosstalk feed poller:")


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
    """The feed's lines, and among HEAD's last 300 commits the "Crosstalk:" ones and the others that change the v3
    folder's data/ or README.md and are not this job's (merges list no files, so they count only by subject)."""
    head = git("rev-parse", "--short=8", "HEAD").strip()
    lines = [ln for ln in show(FEED).splitlines() if ln.strip()]
    commits = []
    for rec in git("log", "-300", "--format=%x00%h%x09%aI%x09%s", "--name-only").split("\x00"):
        rows = [r for r in rec.splitlines() if r.strip()]
        if not rows:
            continue
        h, ad, subj = rows[0].split("\t", 2)
        files = [f for f in rows[1:] if f.startswith(WATCHED)]
        if subj.startswith(PREFIX) or (files and not subj.startswith(MINE)):
            commits.append(dict(hash=h, key=f"{ad} {subj}", subject=subj, date=ad[5:16].replace("T", " "), files=files))
    holds = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "logs").glob("*v3id*/holds.npz")
                   if (m := HOLDS_RE.search(p.parent.name)) and int(m.group(1)) >= 3)
    return dict(head=head, lines=lines, commits=commits, holds=holds)


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
    holds = [h for h in snap["holds"] if h not in set(st.get("holds", []))]
    return dict(lines=lines, rewritten=rewritten, commits=commits, holds=holds)


def report(snap, nw, first):
    print(f"HEAD {snap['head']}; feed {len(snap['lines'])} lines; news first seen {first}")
    if nw["rewritten"]:
        print("the feed's handled lines changed (rewritten or shortened): all lines follow")
    for ln in nw["lines"]:
        print(ln)
    if nw["commits"]:
        print(f'commits not acknowledged ("{PREFIX}", or changing the v3 folder\'s data/ or README.md):')
        for c in nw["commits"]:
            fs = ", ".join(f[len(V3):] for f in c.get("files", [])[:4])
            print(f"  {c['hash']} {c['date']} {c['subject']}" + (f"  [{fs}]" if fs else ""))
    if nw["holds"]:
        print("analysed holds not acknowledged:")
        for h in nw["holds"]:
            print(f"  {h} ({time.strftime('%m-%d %H:%M', time.localtime((ROOT / h).stat().st_mtime))})")


def ack(path, snap):
    st = dict(n_lines=len(snap["lines"]), prefix_sha=sha(snap["lines"]), commit_keys=[c["key"] for c in snap["commits"]],
              holds=snap["holds"], head=snap["head"], acked=time.strftime("%Y-%m-%d %H:%M:%S"))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)
    print(f"acknowledged {st['n_lines']} feed lines, {len(st['commit_keys'])} commits and {len(st['holds'])} holds files "
          f"at HEAD {snap['head']}")


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
        has = bool(nw["lines"] or nw["commits"] or nw["rewritten"] or nw["holds"])
        if a.once:
            if has:
                report(snap, nw, time.strftime("%H:%M:%S"))
                return 0
            print(f"no news at HEAD {snap['head']} ({len(snap['lines'])} feed lines)")
            return 3
        if has:
            sig = (len(snap["lines"]), len(nw["commits"]), len(nw["holds"]), snap["head"])
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
