#!/usr/bin/env python3
"""Held cosine of every checkpoint of the RL contact comparison (2026-10-06): TPU block mesh against 1 mm pads.

For each run of scripts/rl_contact_train_queue.sh and each saved checkpoint (every 41 iterations = 2.0 M env steps,
and the final one), scripts/policy_eval_suite.py rolls the deterministic policy (action = the policy mean) in 64
parallel envs of the run's own scene for 250 policy steps (5 s), with the training's timing (residual and reorient
from step 58, read from the run's config) and no randomisation. Recorded per checkpoint: final cosine of the tool
axis with vertical (mean over the 64), the number of envs still holding the tool at the end (fingertip force above
0.5 N and tool above 60 mm), the number that reached cos 0.9 and held, and the pad forces. One GPU job at a time:
run it after the training queue.

    uv run --extra rl --extra gpu python scripts/rl_contact_eval.py
Rows: docs/experiments/20261006-rl_contact/train_eval.jsonl (fsynced; checkpoints already in it are skipped).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/experiments/20261006-rl_contact/train_eval.jsonl"
JS = ROOT / "logs/20261006-rl_contact/eval"
MORPH = ROOT / "results/phase1/real_v1"
STEPS_PER_IT = 2048 * 24


def runs():
    for variant in ("tpu2.7mesh", "tpu2.7pads1"):
        for seed in (0, 1):
            tag = f"20261006-d6_work_{variant.replace('.', '')}_20M_s{seed}"
            yield tag, variant, seed, ROOT / "results/rl" / tag, MORPH / f"20261006-sv1_u0308_b050_work_tip_{variant}"


def main():
    done = set()
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["tag"], r["iteration"]))
    JS.mkdir(parents=True, exist_ok=True)
    for tag, variant, seed, rd, morph in runs():
        ckpts = sorted((rd / "tensorboard").glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
        for ck in ckpts:
            it = int(re.findall(r"\d+", ck.stem)[0])
            if (tag, it) in done:
                continue
            js = JS / f"{tag}_{it}.json"
            cmd = [sys.executable, str(ROOT / "scripts/policy_eval_suite.py"), "--policy", str(ck),
                   "--morphology-run", str(morph), "--n", "64", "--steps", "250", "--align-thresh", "0.9",
                   "--lift-delta", "0.1", "--finger-residual-scale", "0.5", "--open-finger-from-keyframe",
                   "--closed-ctrl-from-keyframe", "open_ik", "--json-out", str(js), "--label", f"{tag} it {it}"]
            env = dict(os.environ, MUJOCO_GL="egl")
            w0 = time.perf_counter()
            p = subprocess.run(cmd, capture_output=True, text=True, env=env)
            row = {"tag": tag, "variant": variant, "seed": seed, "iteration": it, "env_steps": (it + 1) * STEPS_PER_IT,
                   "checkpoint": str(ck.relative_to(ROOT)), "wall_s": round(time.perf_counter() - w0, 1),
                   "when": time.strftime("%Y-%m-%d %H:%M")}
            if p.returncode == 0 and js.exists():
                row.update(json.loads(js.read_text()), status="ok")
            else:
                row.update(status="error", error=(p.stderr or p.stdout)[-1500:])
            with open(OUT, "a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            print(f"{tag} it {it}: " + (f"final cos {row['final_cos_mean']:+.3f}, held {row['hold_rate'] * 64:.0f}/64, "
                  f"aligned+held {row['success_rate'] * 64:.0f}/64" if row["status"] == "ok" else row["error"][-300:]),
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
