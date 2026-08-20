#!/usr/bin/env python3
"""Create, inspect and resume packed DeltaAI work queues.

A queue is a directory holding a manifest plus the claim/done/failed state that
`deltaai_pack_queue.slurm` writes as it drains it. Because the state is on disk
rather than in Slurm, "where did we leave off" is a question you can answer
without the scheduler, and the next block resumes with no bookkeeping.

  # create from a TSV on stdin (EST_HOURS <TAB> LABEL <TAB> COMMAND)
  scripts/cluster/deltaai_queue.py new inline_seeds < tasks.tsv

  # what is left, and how many SU it will take
  scripts/cluster/deltaai_queue.py status inline_seeds

  # how many 8-hour workers to submit to finish in one pass
  scripts/cluster/deltaai_queue.py plan inline_seeds --block-hours 8

  # clear failures so the next block retries them
  scripts/cluster/deltaai_queue.py retry inline_seeds
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def qdir(name: str) -> Path:
    return ROOT / "queue" / name


def read_manifest(q: Path) -> list[tuple[float, str, str]]:
    out = []
    for ln in (q / "manifest.tsv").read_text().splitlines():
        if not ln.strip():
            continue
        est, label, cmd = ln.split("\t", 2)
        out.append((float(est), label, cmd))
    return out


def cmd_new(args) -> int:
    q = qdir(args.name)
    if q.exists() and not args.force:
        print(f"!! {q} exists; pass --force to replace it", file=sys.stderr)
        return 1
    if q.exists():
        shutil.rmtree(q)
    for sub in ("", "claimed", "done", "failed", "logs"):
        (q / sub).mkdir(parents=True, exist_ok=True)

    lines = [ln for ln in sys.stdin.read().splitlines() if ln.strip()]
    bad = [i for i, ln in enumerate(lines, 1) if ln.count("\t") < 2]
    if bad:
        print(f"!! lines {bad} are not EST_HOURS<TAB>LABEL<TAB>COMMAND", file=sys.stderr)
        return 1
    (q / "manifest.tsv").write_text("\n".join(lines) + "\n")

    tasks = read_manifest(q)
    total = sum(t[0] for t in tasks)
    print(f"queue '{args.name}': {len(tasks)} tasks, {total:.1f} GPU-hours of work")
    print(f"  sbatch --time=08:00:00 scripts/cluster/deltaai_pack_queue.slurm {args.name}")
    return 0


def cmd_status(args) -> int:
    q = qdir(args.name)
    if not q.is_dir():
        print(f"!! no such queue: {q}", file=sys.stderr)
        return 1
    tasks = read_manifest(q)
    done = {int(p.name) for p in (q / "done").iterdir()} if (q / "done").is_dir() else set()
    failed = {int(p.name) for p in (q / "failed").iterdir()} if (q / "failed").is_dir() else set()
    claimed = {int(p.name) for p in (q / "claimed").iterdir()} if (q / "claimed").is_dir() else set()

    todo = [i for i in range(1, len(tasks) + 1) if i not in done and i not in failed]
    left = sum(tasks[i - 1][0] for i in todo)
    print(f"queue '{args.name}': {len(tasks)} tasks")
    print(f"  done     {len(done)}")
    print(f"  failed   {len(failed)}{'  -> ' + ','.join(map(str, sorted(failed))) if failed else ''}")
    print(f"  running  {len(claimed)}{'  -> ' + ','.join(map(str, sorted(claimed))) if claimed else ''}")
    print(f"  todo     {len(todo)}  ({left:.1f} GPU-hours remaining)")
    if args.verbose:
        for i, (est, label, _) in enumerate(tasks, 1):
            state = ("done" if i in done else "FAILED" if i in failed
                     else "running" if i in claimed else "todo")
            print(f"    {i:3d} {state:8s} {est:4.1f}h  {label}")
    return 0


def cmd_plan(args) -> int:
    q = qdir(args.name)
    tasks = read_manifest(q)
    done = {int(p.name) for p in (q / "done").iterdir()} if (q / "done").is_dir() else set()
    failed = {int(p.name) for p in (q / "failed").iterdir()} if (q / "failed").is_dir() else set()
    todo = [tasks[i - 1] for i in range(1, len(tasks) + 1) if i not in done and i not in failed]
    left = sum(t[0] for t in todo)
    longest = max((t[0] for t in todo), default=0.0)
    b = args.block_hours

    if longest > b:
        print(f"!! the longest remaining task is {longest:.1f}h but the block is {b:.0f}h — "
              f"it can never start. Raise --block-hours.", file=sys.stderr)
        return 1

    # Packing waste is the tail of each block: on average half a task's length.
    workers = max(1, round(left / (b * 0.85) + 0.5))
    print(f"{left:.1f} GPU-hours remain, longest task {longest:.1f}h")
    print(f"  {workers} worker(s) x {b:.0f}h blocks = {workers * b:.0f} SU reserved "
          f"(~{workers * b - left:.1f} SU of tail waste)")
    print(f"  wall clock if all {workers} run concurrently: ~{left / workers:.1f} h")
    print()
    print(f"  for i in $(seq {workers}); do sbatch --time={int(b):02d}:00:00 "
          f"scripts/cluster/deltaai_pack_queue.slurm {args.name}; done")
    return 0


def cmd_retry(args) -> int:
    q = qdir(args.name)
    n = 0
    for p in (q / "failed").iterdir():
        p.unlink()
        n += 1
    print(f"cleared {n} failure(s); the next block will retry them")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("new", help="create a queue from a TSV on stdin")
    p.add_argument("name")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("status", help="done / running / todo, and hours left")
    p.add_argument("name")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("plan", help="how many workers to submit")
    p.add_argument("name")
    p.add_argument("--block-hours", type=float, default=8.0)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("retry", help="clear failures so they run again")
    p.add_argument("name")
    p.set_defaults(func=cmd_retry)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
