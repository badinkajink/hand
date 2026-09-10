"""Score a set of real_v1 designs with KaRMA, in parallel, resumably.

Each (design, finger pair, variant) is one subprocess running the upstream
`run_metric.py` unmodified, with its own generated URDF, robot YAML and metric config so
that runs cannot collide on `workspace/current.yaml`. Results append to a JSONL, and a
re-run skips whatever is already in it.

Two variants:

  scaled     KaRMA as published. Every length in `karma_config.yaml` is multiplied by
             L_ref/200 mm, so each hand is posed a geometrically equivalent problem and
             the score is exactly invariant to uniform rescaling of the hand.
  absolute   the same search with the scaling defeated, by setting `l_ref_nominal_m` to
             the hand's own L_ref so the factor is exactly 1, and with the test object and
             finger radius pinned to the hardware: a 12.5 mm sphere (the screwdriver
             shaft's radius) and a 10.55 mm capsule (the real pad). This asks what the
             hand can do to OUR object rather than to a hand-sized one.

Usage:

    python scripts/karma_batch.py --sample docs/experiments/.../sample.json \\
        --karma-root external/karma --out .../karma_scores.jsonl --jobs 14
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE_MJCF = ROOT / "assets/mjcf/real_v1/real_hand.xml"
PAIRS = ("thumb-index", "thumb-middle", "index-middle")

# The hardware the absolute variant pins to (assets/mjcf/real_v1/*): the screwdriver shaft
# is a cylinder of radius 12.5 mm and the finger pad a sphere of radius 10.55 mm.
SHAFT_RADIUS_M = 0.0125
PAD_RADIUS_M = 0.01055

# karma.seed_selection sizes its own ProcessPoolExecutor from the physical core count,
# independently of `rotation_workers`, so N concurrent runs become ~10N processes and the
# machine thrashes. This wrapper pins that pool to one process so the batch can schedule
# one run per core. Verified bit-identical on the base hand: 46 voxels, KaRMA-T 0.005667,
# KaRMA-R 0.1652, KaRMA-S 0.4348 either way, which is what the upstream README promises
# ("worker count affects speed only, not the result").
SERIAL_WRAPPER = '''#!/usr/bin/env python3
"""run_metric.py with KaRMA's internal seed-selection pool pinned to one process."""
import runpy

import karma.seed_selection as _ss

_ss._physical_core_count = lambda: 1
runpy.run_path("run_metric.py", run_name="__main__")
'''


def _run_one(job: dict) -> dict:
    karma = Path(job["karma_root"])
    work = Path(job["work"])
    tag = f"{job['design']}_{job['pair'].replace('-', '_')}_{job['variant']}"
    rdir = work / tag
    rdir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONPATH", None)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = "1"

    mounts = ",".join(f"{x:.9f}" for x in job["mounts_mm"])
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/karma_export_urdf.py"),
         "--mjcf", str(BASE_MJCF), f"--mounts={mounts}",
         "--name", job["design"], "--pair", job["pair"], "--out-dir", str(rdir)],
        check=True, capture_output=True)

    robot_cfg = rdir / f"robot_{job['design']}_{job['pair'].replace('-', '_')}.yaml"
    metric = yaml.safe_load((karma / "karma_config.yaml").read_text())
    metric["out_dir"] = str(rdir.resolve())
    metric["rotation_workers"] = 1

    if job["variant"] == "absolute":
        sys.path.insert(0, str(karma))
        from karma.lref import compute_lref  # noqa: PLC0415  (needs the karma env on sys.path)
        metric["l_ref_nominal_m"] = float(compute_lref(robot_cfg))
        metric["sphere_radius_m"] = SHAFT_RADIUS_M
        metric["link_radius_m"] = PAD_RADIUS_M

    mpath = rdir / "metric.yaml"
    mpath.write_text(yaml.safe_dump(metric))

    t0 = time.time()
    proc = subprocess.run(
        [str(Path(job["python"])), "run_metric_serial.py",
         "--config", str(robot_cfg.resolve()), "--metric-config", str(mpath.resolve()),
         "--rotation-workers", "1"],
        cwd=str(karma), env=env, capture_output=True, text=True, timeout=1800)

    rec = {"design": job["design"], "pair": job["pair"], "variant": job["variant"],
           "secs": round(time.time() - t0, 1)}
    res = rdir / "current.yaml"
    if proc.returncode != 0 or not res.exists():
        rec["error"] = (proc.stderr or proc.stdout or "no output")[-400:]
        return rec
    y = yaml.safe_load(res.read_text())
    # results.py nests the scalars under `summary`, not at the top level.
    sm = y.get("summary") or {}
    ss = y.get("seed_sensitivity") or {}
    rec.update({
        "karma_t": sm.get("translational_score"),
        "karma_r": sm.get("global_rotational_score"),
        "karma_s": ss.get("karma_s"),
        "n_voxels": sm.get("n_voxels_reached"),
        "n_states": sm.get("n_states_reached"),
        "l_ref_mm": round(float(sm.get("l_ref_m", 0.0)) * 1e3, 3),
        "volume_m3": sm.get("translational_volume_m3"),
        "n_seeds": ss.get("n_seeds_evaluated"),
    })
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--karma-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--python", default=str(Path.home() / "miniconda3/envs/karma-hand-metric/bin/python"))
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--variants", default="scaled")
    ap.add_argument("--limit", type=int, default=0, help="first N designs only (smoke test)")
    args = ap.parse_args()

    rows = json.loads(args.sample.read_text())
    if args.limit:
        rows = rows[:args.limit]

    done: set[tuple] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" not in r:
                done.add((r["design"], r["pair"], r["variant"]))

    jobs = []
    for r in rows:
        for pair in args.pairs.split(","):
            for variant in args.variants.split(","):
                if (r["design"], pair, variant) in done:
                    continue
                jobs.append({"design": r["design"], "mounts_mm": r["mounts_mm"],
                             "pair": pair, "variant": variant,
                             "karma_root": str(args.karma_root.resolve()),
                             "work": str(args.work.resolve()), "python": args.python})

    wrapper = args.karma_root / "run_metric_serial.py"
    if not wrapper.exists() or wrapper.read_text() != SERIAL_WRAPPER:
        wrapper.write_text(SERIAL_WRAPPER)

    print(f"{len(rows)} designs, {len(done)} runs already done, {len(jobs)} to run "
          f"on {args.jobs} workers", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_err = 0
    with args.out.open("a") as fh, ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(_run_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as exc:  # a crashed worker must not kill the sweep
                j = futs[fut]
                rec = {"design": j["design"], "pair": j["pair"], "variant": j["variant"],
                       "error": f"{type(exc).__name__}: {exc}"[:400]}
            n_err += "error" in rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(jobs):
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  {el / 60:.1f} min elapsed, "
                      f"~{el / i * (len(jobs) - i) / 60:.1f} min left, {n_err} errors",
                      flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min, {n_err} errors -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
