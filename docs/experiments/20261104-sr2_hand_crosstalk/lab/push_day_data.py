"""Push 11-05's crosstalk data to a branch without touching the working tree, the index or main: the board log cut to
[T0, T1] by t (fixed 248-byte records), and each run's events, telemetry and progress, as LFS objects written straight
into .git/lfs/objects at a capped rate (corvus's disk also holds the live board log; a burst of writes stalled the logger
at 18:09 on 11-05), then pointer blobs in a temporary index, one commit on PARENT, the LFS objects pushed, then the
commit to BRANCH. The layout is the 11-04 night's (data/20261104-night): boards/<time>-mux.bin and .json, one folder per
run, chain5's chunks as chain5/<chunk>/.

    /usr/bin/python3 push_day_data.py --t0 HH:MM:SS --t1 HH:MM:SS --parent REV --branch NAME [--rate MB/s] [--dry] \\
        --runs RUN_DIR... --outs FILE... --note TEXT...
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/will/workspace/hand")
LOGS = REPO / "logs/20261105-sr2_crosstalk"
DEST = "docs/experiments/20261104-sr2_hand_crosstalk/data/20261105-day"
BIN = LOGS / "boards/20261105-114116-mux.bin"
LFS = REPO / ".git/lfs"
ap = argparse.ArgumentParser()
ap.add_argument("--t0", required=True)
ap.add_argument("--t1", required=True)
ap.add_argument("--parent", required=True)
ap.add_argument("--branch", required=True)
ap.add_argument("--rate", type=float, default=4.0, help="MB/s written at most, from the first write on")
ap.add_argument("--runs", nargs="*", default=[], help="run folders relative to the day's logs")
ap.add_argument("--outs", nargs="*", default=[], help="text logs relative to the day's logs (plain git files)")
ap.add_argument("--note", nargs="*", default=[], help="lines for the manifest's 'uncertain'")
ap.add_argument("--message", default=None)
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
at = lambda hms: time.mktime(time.strptime("2026-11-05 " + hms, "%Y-%m-%d %H:%M:%S"))
T0, T1 = at(a.t0), at(a.t1)
start, written = time.time(), 0


def git(*args, data=None, env=None):
    r = subprocess.run(["git", *args], cwd=REPO, input=data, capture_output=True, check=True, env=env)
    return r.stdout.decode().strip()


def store(chunks):
    """The chunks into the LFS store at the capped rate, synced every 32 MB; (oid, size)."""
    global written
    tmp = LFS / "tmp" / f"day-{os.getpid()}-{time.time_ns()}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    h, size, unsynced = hashlib.sha256(), 0, 0
    with open(tmp, "wb") as f:
        for c in chunks:
            f.write(c)
            h.update(c)
            size += len(c)
            unsynced += len(c)
            written += len(c)
            ahead = written / (a.rate * 1e6) - (time.time() - start)
            if ahead > 0:
                time.sleep(ahead)
            if unsynced >= 8 << 20:
                f.flush()
                os.fsync(f.fileno())
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                unsynced = 0
        f.flush()
        os.fsync(f.fileno())
    oid = h.hexdigest()
    dst = LFS / "objects" / oid[:2] / oid[2:4] / oid
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        tmp.unlink()
    else:
        os.rename(tmp, dst)
    return oid, size


def chunks_of(path, lo=0, hi=None, bs=1 << 20):
    with open(path, "rb") as f:
        f.seek(lo)
        left = None if hi is None else hi - lo
        while left is None or left > 0:
            c = f.read(bs if left is None else min(bs, left))
            if not c:
                return
            yield c
            os.posix_fadvise(f.fileno(), max(0, f.tell() - len(c)), len(c), os.POSIX_FADV_DONTNEED)
            if left is not None:
                left -= len(c)


entries = []


def add_lfs(path, chunks):
    oid, size = store(chunks)
    blob = git("hash-object", "-w", "--stdin",
               data=f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n".encode())
    entries.append((path, blob, oid, size))
    print(f"{path}: {size / 1e6:.1f} MB, lfs {oid[:12]}", flush=True)


def add_plain(path, data):
    entries.append((path, git("hash-object", "-w", "--stdin", data=data), None, len(data)))
    print(f"{path}: {len(data) / 1e3:.1f} kB, plain", flush=True)


# the board log, cut by t
side = json.loads(BIN.with_suffix(".json").read_text())
nb = len(side["channels"])
rec = 8 + 80 * nb
m = np.memmap(BIN, dtype=[("t", "<f8"), ("r", "<f4", (20 * nb,))], mode="r")
t = m["t"]                                 # a strided view: the search reads a few pages, not the whole file
i0, i1 = int(np.searchsorted(t, T0, "left")), int(np.searchsorted(t, T1, "right"))
name = time.strftime("%Y%m%d-%H%M%S", time.localtime(T0)) + "-mux"
cut = dict(side, start=float(t[i0]), cut_of=BIN.name, cut_t0=T0, cut_t1=T1, records=i1 - i0,
           first=float(t[i0]), last=float(t[i1 - 1]))
print(f"boards: records {i0} to {i1 - 1} of {BIN.name}, {time.strftime('%H:%M:%S', time.localtime(t[i0]))} to "
      f"{time.strftime('%H:%M:%S', time.localtime(t[i1 - 1]))}, {(i1 - i0) * rec / 1e6:.0f} MB", flush=True)
runs = []
for r in a.runs:
    d = LOGS / r
    ev = [json.loads(x) for x in open(d / "events.jsonl")]
    runs.append({"run": r, "first": ev[0]["t"], "last": ev[-1]["t"], "steps": sum(e["kind"] == "step" for e in ev),
                 "end": next((e.get("reason") for e in reversed(ev) if e["kind"] == "end"), None),
                 "files": sorted(p.name for p in d.iterdir() if p.name in ("events.jsonl", "telemetry.jsonl",
                                                                           "progress.json"))})
manifest = {"what": "SR2 hand crosstalk data of 11-05 from corvus: the fingertip boards' log (cut by t) and the runner's "
                    "events, telemetry and progress of each run; every time is corvus's clock (time.time())",
            "boards": dict(cut, file=f"boards/{name}.bin"), "runs": runs, "uncertain": a.note,
            "telemetry_hz": "about 8.6: run_plan.py polls the station every 0.1 s plus the request (the station "
                            "samples at 20 Hz)",
            "board_records": "248 bytes: t <f8 (corvus's time at receipt), then per board 20 float32 (5 chips x "
                             "[temperature, bx, by, bz] in uT); about 2.6 misframed records an hour hold values like "
                             "1e23 (drop non-finite or |value| > 2e4)"}
if a.dry:
    print(json.dumps(manifest, indent=1)[:3000])
    raise SystemExit(0)
start, written = time.time(), 0            # the cap counts from the first write (18:28 on 11-05: it counted from
# the script's start, so the time spent before let 240 MB out at full speed and stalled the board logger)
add_lfs(f"{DEST}/boards/{name}.bin", chunks_of(BIN, i0 * rec, i1 * rec))
add_lfs(f"{DEST}/boards/{name}.json", iter([json.dumps(cut, indent=1).encode()]))
for r in a.runs:
    for f in ("events.jsonl", "telemetry.jsonl", "progress.json"):
        if (LOGS / r / f).exists():
            add_lfs(f"{DEST}/{r}/{f}", chunks_of(LOGS / r / f))
for o in a.outs:
    add_plain(f"{DEST}/{Path(o).name}", (LOGS / o).read_bytes())
add_lfs(f"{DEST}/manifest_{a.t0.replace(':', '')}_{a.t1.replace(':', '')}.json",
        iter([json.dumps(manifest, indent=1).encode()]))
env = dict(os.environ, GIT_INDEX_FILE=str(REPO / ".git" / "push_day_data.idx"))
git("read-tree", a.parent, env=env)
for path, blob, _, _ in entries:
    git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
tree = git("write-tree", env=env)
msg = a.message or f"SR2 crosstalk data of 11-05, {a.t0} to {a.t1}"
commit = git("commit-tree", tree, "-p", a.parent, "-m", msg)
print(f"commit {commit[:9]} on {a.parent[:9]}, {len(entries)} files, {written / 1e6:.0f} MB written in "
      f"{time.time() - start:.0f} s", flush=True)
oids = [e[2] for e in entries if e[2]]
subprocess.run(["git", "lfs", "push", "--object-id", "origin", *oids], cwd=REPO, check=True)
subprocess.run(["git", "push", "origin", f"{commit}:refs/heads/{a.branch}"], cwd=REPO, check=True)
print(f"pushed {commit[:9]} to {a.branch}", flush=True)
