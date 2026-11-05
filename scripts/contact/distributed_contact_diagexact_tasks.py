#!/usr/bin/env python3
"""MuJoCo 3.14 hand-task ablation: exact diagonal versus impedance mapping.

Use the current-engine venv. Every row keeps the same SR2, tool, controller,
friction, and timestep within its task. The physical policy also changes the
contact regularizer, so compare it with its own diagexact/runtime variants;
the legacy policy supplies the isolated flag-only contrast.
"""

import argparse
import json

import mujoco

from contact_surface.records import ROOT, versions, write_json
from distributed_contact_tasks import OUT, run_case

SPEC = "mj:spheres:s1:rs0.75:tr0.03:ir100"
POLICIES = (
    ("legacy", "legacy"),
    ("legacy_exact", "legacy_diagexact"),
    ("physical", "physical"),
    ("physical_exact", "physical_diagexact"),
    ("physical_runtime_approx", "physical_runtime_approx"),
    ("physical_runtime_exact", "physical_runtime_diagexact"),
)


def cases():
    for task in ("chain", "roll"):
        for short, policy in POLICIES:
            yield dict(
                id=f"v314_{task}_{short}_s1_50us",
                spec=SPEC,
                contact_policy=policy,
                physics_dt=0.00005,
                **({"seed": 0} if task == "chain" else {"controller_hz": 500}),
                film=short in {"legacy_exact", "physical_exact", "physical_runtime_exact"},
            )
    for task, ratio in (("chain", 100), ("roll", 100), ("roll", 1000)):
        yield dict(
            id=f"v314_{task}_physical_runtime_ct{ratio}_exact_s1_50us",
            spec=SPEC,
            contact_policy=f"physical_runtime_ct{ratio}_diagexact",
            physics_dt=0.00005,
            **({"seed": 0} if task == "chain" else {"controller_hz": 500}),
            film=ratio == 100,
        )


def main():
    if tuple(int(x) for x in mujoco.__version__.split(".")[:2]) < (3, 9):
        raise RuntimeError("diagexact requires MuJoCo >= 3.9")
    all_cases = list(cases())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=[c["id"] for c in all_cases])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    write_json(OUT / "versions-v314.json", versions())
    selected = [c for c in all_cases if args.case is None or c["id"] in args.case]
    results = []
    for case in selected:
        path = OUT / case["id"] / "summary.json"
        if path.exists() and not args.force:
            result = json.loads(path.read_text())
            print(case["id"], "cached", result["status"], flush=True)
        else:
            result = run_case(case)
        results.append(result)
    if args.case is None:
        write_json(ROOT / "docs/experiments/20261004-codex/data/task_diagexact_results.json", results)


if __name__ == "__main__":
    main()
