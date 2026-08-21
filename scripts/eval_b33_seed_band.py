#!/usr/bin/env python3
"""Score the b33 seed band: where does b33 sit in the distribution it was drawn from?

Every inline sim2real conclusion — the three robustness cliffs, the fingertip
ranking, the hardware spec (mu >= 1.7, 1-3 mm pad deflection, r 5-6 mm convex tip)
— traces back to ONE policy draw, b33, evaluated at n=32 rollouts. The n=32
ROLLOUT spread is characterised at ~+-0.1. The TRAINING-draw spread never was, and
elsewhere in this program it runs 0.3-0.5, larger than every difference those
tables rank.

This scores the 16 cluster draws plus b33 itself through the continuous handoff —
`rl_demo_handoff_continuous.py`, NOT `policy_eval_suite`, which measures the wrong
thing for this arrangement (two_arrangements finding).

Two variances are separable here and must not be conflated: WITHIN a draw (GPU
contact solves are non-deterministic, so the same checkpoint ends differently) and
BETWEEN draws (the thing being measured). --repeats samples the first so the
second is not read off a single rollout.

  uv run --extra rl --extra gpu python scripts/eval_b33_seed_band.py
  uv run ... --repeats 3 --runs 'results/rl/20260820-21*b33seed_s*'
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MORPH = ROOT / "results/phase1/landscape/m05_ik_cem"
A_CKPT = ROOT / "results/rl/a10_20260702-1256-policyA_m05_ik10/tensorboard/model_609.pt"
B33 = ROOT / "results/rl/b33_20260702-1353-policyB_m05_reorient_ik10/tensorboard/model_270.pt"
OUT_J = ROOT / "docs/experiments/B33_SEED_BAND.json"
OUT_T = ROOT / "docs/experiments/B33_SEED_BAND.txt"

RE_COS = re.compile(r"held-vertical cos POST-HANDOFF \(last 50 steps mean\): ([-\d.]+) \(peak ([-\d.]+)\)")
RE_MINZ = re.compile(r"min object-center z POST-HANDOFF[^:]*: ([\d.]+)")


def one(ckpt: Path, workdir: Path) -> dict:
    import os
    env = dict(os.environ)
    env["WARP_CACHE_PATH"] = tempfile.mkdtemp()
    env["MUJOCO_GL"] = "egl"
    cmd = [
        "python", "scripts/rl_demo_handoff_continuous.py",
        "--policy-a", str(A_CKPT), "--policy-b", str(ckpt),
        "--morphology-run", str(MORPH),
        "--open-finger-from-keyframe",
        "--handoff-step", "40", "--lift-delta", "0.10", "--total-steps", "240",
        "--output", str(workdir / f"{ckpt.parent.parent.name}.mp4"),
    ]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    txt = p.stdout + p.stderr
    m, z = RE_COS.search(txt), RE_MINZ.search(txt)
    return {
        "held_cos": float(m.group(1)) if m else None,
        "peak_cos": float(m.group(2)) if m else None,
        "minz_post": float(z.group(1)) if z else None,
        "rc": p.returncode,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/rl/20260820-2*b33seed_s*",
                    help="glob of run dirs (the CORRECTED batch; the 1714 run used the "
                         "wrong b10 warmstart and must not be included)")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    work = ROOT / "logs" / f"{time.strftime('%Y%m%d-%H%M')}-b33_seed_band"
    work.mkdir(parents=True, exist_ok=True)

    targets: list[tuple[str, Path]] = []
    for d in sorted(glob.glob(str(ROOT / args.runs))):
        c = Path(d) / "tensorboard/model_270.pt"
        if c.is_file():
            targets.append((Path(d).name.split("-")[-1], c))
    targets.append(("b33_REFERENCE", B33))

    recs = json.loads(OUT_J.read_text()) if OUT_J.exists() else []
    seen = {r["name"] for r in recs}

    for name, ckpt in targets:
        if name in seen:
            print(f"[skip] {name}")
            continue
        trials = [one(ckpt, work) for _ in range(args.repeats)]
        cos = [t["held_cos"] for t in trials if t["held_cos"] is not None]
        rec = {
            "name": name, "ckpt": str(ckpt.relative_to(ROOT)),
            "trials": trials,
            "cos_mean": round(statistics.fmean(cos), 3) if cos else None,
            "cos_sd": round(statistics.stdev(cos), 3) if len(cos) > 1 else None,
            "held": sum(1 for t in trials if (t["minz_post"] or 0) > 0.05),
            "n": len(trials),
        }
        recs.append(rec)
        OUT_J.write_text(json.dumps(recs, indent=2))
        print(f"[{name}] cos {rec['cos_mean']} +-{rec['cos_sd']}  held {rec['held']}/{rec['n']}")

    draws = [r for r in recs if r["name"] != "b33_REFERENCE" and r["cos_mean"] is not None]
    ref = next((r for r in recs if r["name"] == "b33_REFERENCE"), None)
    means = sorted(r["cos_mean"] for r in draws)

    lines = [f"# b33 seed band {time.strftime('%Y-%m-%d %H:%M')} — {len(draws)} draws, "
             f"a10 warmstart, 20M ts, m05, continuous handoff, {args.repeats} rollouts/draw", ""]
    for r in sorted(draws, key=lambda x: -x["cos_mean"]):
        lines.append(f"{r['name']:16} cos {r['cos_mean']:+.3f} "
                     f"sd {str(r['cos_sd']):>6}  held {r['held']}/{r['n']}")
    if ref:
        lines += ["", f"{'b33 (REFERENCE)':16} cos {ref['cos_mean']:+.3f} "
                      f"sd {str(ref['cos_sd']):>6}  held {ref['held']}/{ref['n']}"]
    if means:
        lines += ["",
                  f"draw band: min {means[0]:+.3f}  median {statistics.median(means):+.3f}  "
                  f"max {means[-1]:+.3f}",
                  f"between-draw sd: {statistics.stdev(means):.3f}" if len(means) > 1 else "",
                  f"within-draw sd (mean over draws): "
                  f"{statistics.fmean([r['cos_sd'] for r in draws if r['cos_sd'] is not None]):.3f}"
                  if any(r["cos_sd"] is not None for r in draws) else ""]
        if ref and ref["cos_mean"] is not None:
            better = sum(1 for m in means if m > ref["cos_mean"])
            lines.append(f"b33 percentile: {len(means) - better}/{len(means)} draws at or below it")
    OUT_T.write_text("\n".join(l for l in lines if l is not None) + "\n")
    print()
    print(OUT_T.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
