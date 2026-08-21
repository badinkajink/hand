#!/usr/bin/env python3
"""Where is the NaN cliff on the m05 inline scene, as a function of init_noise_std?

Why this exists. The 16-seed b33 study on DeltaAI returned 1 completion and 15
`observation group 'actor' contains NaN` failures, at iterations 2, 2, 2, 7, 7, 8,
10, 19, 27, 27, 44, 92, 143, 180, 185 — a per-step hazard, not a bad-seed cluster.
Three of those seeds were re-run locally and NaN'd there too (iters 21, 36, 22), so
it is the configuration, not the aarch64 platform.

b33's own config carries `init_noise_std: 0.3` (the trainer default; b_liveA does
NOT pin it). The perp recipe pins 0.05 with a measured table for exactly this
failure — 0.05 clean, 0.15 NaN@227, 0.5 NaN@14, 1.0 NaN@12 — and the note that
"the perp scene NaNs above ~0.1". Nobody ever ran that measurement on the m05
inline scene, because b33 was the only draw ever taken on it at 0.3.

So the working hypothesis is that b33 trained on the unstable side of a known
cliff and survived a ~6% event. This probe tests it and locates the safe setting,
which decides whether the seed study can be re-run at all and at what value.

Cheap by construction: NaN strikes early (11 of 15 failures inside 30 iterations),
so short runs resolve it. Runs LOCALLY — it costs no allocation.

  uv run --extra rl --extra gpu python scripts/probe_b33_noise_stability.py
  uv run ... --stds 0.15 0.1 0.05 --seeds 9 13 10 --timesteps 3000000
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MORPH = "results/phase1/landscape/m05_ik_cem"
A_CKPT = "results/rl/a10_20260702-1256-policyA_m05_ik10/tensorboard/model_609.pt"
B10 = "results/rl/b10_20260604-1642-policyB_holdonlyws_repro/tensorboard/model_541.pt"
OUT = ROOT / "docs/experiments/B33_NOISE_STABILITY.json"
TXT = ROOT / "docs/experiments/B33_NOISE_STABILITY.txt"


def one(std: float, seed: int, timesteps: int, logdir: Path) -> dict:
    """One short live-A run. Returns iterations survived and whether it NaN'd."""
    tag = f"noiseprobe_s{seed}_std{std}"
    log = logdir / f"{tag}.log"
    cmd = [
        "python", "scripts/rl_train_cube.py",
        "--recipe", "b_liveA",
        "--morphology-run", MORPH,
        "--num-envs", "3072", "--total-timesteps", str(timesteps),
        "--live-a-checkpoint", A_CKPT, "--live-a-onset", "40", "--live-a-blend-steps", "0",
        "--lift-target-z-above-init", "0.1", "--lift-delta-z", "0.1",
        "--finger-residual-scale", "0.5", "--finger-close-easing", "ease_out_quad",
        # b33's own timing — see make_inline_breadth_queue.py. The ONLY axis this
        # probe moves is init_noise_std; everything else stays at b33's values so a
        # difference is attributable to the noise and nothing else.
        "--lift-phase-start-step", "58", "--reorient-start-step", "58",
        "--term-tip-lost-steps", "10",
        "--target-axis-weight", "100.0", "--target-axis-progress-weight", "300.0",
        "--open-finger-from-keyframe",
        "--init-actor-checkpoint", B10,
        "--init-noise-std", str(std),
        "--seed", str(seed), "--tag", tag,
        "--no-wandb", "--no-record-videos",
    ]
    env = {"WARP_CACHE_PATH": tempfile.mkdtemp(), "MUJOCO_GL": "egl"}
    import os
    e = dict(os.environ); e.update(env)
    t0 = time.time()
    with log.open("w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=e).returncode
    text = log.read_text(errors="replace")
    iters = len(re.findall(r"Iteration time", text))
    nan = "contains NaN" in text
    return {"std": std, "seed": seed, "rc": rc, "iters": iters, "nan": nan,
            "secs": round(time.time() - t0)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stds", nargs="+", type=float, default=[0.3, 0.15, 0.1, 0.05])
    ap.add_argument("--seeds", nargs="+", type=int, default=[9, 13, 10])
    ap.add_argument("--timesteps", type=int, default=3_000_000)
    args = ap.parse_args()

    logdir = ROOT / "logs" / f"{time.strftime('%Y%m%d-%H%M')}-b33_noise_probe"
    logdir.mkdir(parents=True, exist_ok=True)
    recs: list[dict] = []
    if OUT.exists():
        recs = json.loads(OUT.read_text())
    seen = {(r["std"], r["seed"]) for r in recs}

    for std in args.stds:
        for seed in args.seeds:
            if (std, seed) in seen:
                print(f"[skip] std={std} seed={seed}")
                continue
            r = one(std, seed, args.timesteps, logdir)
            recs.append(r)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(recs, indent=2))
            print(f"[std {std} seed {seed}] iters={r['iters']} nan={r['nan']} "
                  f"rc={r['rc']} {r['secs']}s")

    lines = [f"# b33 / m05 inline scene: NaN cliff vs init_noise_std "
             f"{time.strftime('%Y-%m-%d %H:%M')}",
             f"# {args.timesteps} ts per point (~{args.timesteps // 73728} iters if clean); "
             f"b33 trained at std 0.3", ""]
    for std in args.stds:
        pts = [r for r in recs if r["std"] == std]
        if not pts:
            continue
        nans = sum(1 for r in pts if r["nan"])
        iters = ",".join(str(r["iters"]) for r in pts)
        lines.append(f"init_noise_std {std:<5}  NaN {nans}/{len(pts)}   iters survived: {iters}")
    TXT.write_text("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))
    print(f"\n-> {TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
