#!/usr/bin/env python3
"""Isolated newer-MuJoCo flag, mass, acceleration and controlled-slip follow-up."""

import argparse
import contextlib
import json
import traceback
from pathlib import Path

import mujoco

from contact_surface.records import ROOT, versions, write_json
from contact_surface.slip import run as slip_run
from distributed_contact import DOC, expand, run_id, static_run, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["diagexact", "slip"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    configs = expand(args.config or DOC / "configs" / f"{args.phase}.json", args.phase)
    manifest = [dict(run_id=run_id(c), config=c) for c in configs]
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        print(f"{len(manifest)} runs")
        return
    if any(c["engine_version"] != mujoco.__version__ for c in configs):
        raise ValueError(f"Config requests another MuJoCo version; running {mujoco.__version__}")
    out = args.out or ROOT / "results/20261004-distributed-contact" / args.phase
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Manifest differs; use another --out to preserve results")
    write_json(manifest_path, manifest)
    metadata, launched = versions(), 0
    for i, item in enumerate(manifest):
        directory = out / item["run_id"]
        summary_path = directory / "summary.json"
        if (
            summary_path.exists()
            and not args.force
            and json.loads(summary_path.read_text())["status"] == "complete"
        ):
            continue
        if args.limit is not None and launched >= args.limit:
            break
        directory.mkdir(exist_ok=True)
        write_json(directory / "config.json", item["config"])
        write_json(directory / "versions.json", metadata)
        try:
            with (
                (directory / "stdout.log").open("w") as log,
                contextlib.redirect_stdout(log),
                contextlib.redirect_stderr(log),
            ):
                summary = (static_run if args.phase == "diagexact" else slip_run)(
                    item["config"], directory
                )
        except Exception as exc:
            summary = dict(status="failed", error=repr(exc), traceback=traceback.format_exc())
        write_json(summary_path, summary)
        launched += 1
        print(f"{i + 1}/{len(manifest)} {item['run_id']} {summary['status']}", flush=True)
    rows = summarize(out, manifest)
    print(
        {
            state: sum(r["status"] == state for r in rows)
            for state in ["complete", "failed", "missing"]
        }
    )
    if any(r["status"] == "failed" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
