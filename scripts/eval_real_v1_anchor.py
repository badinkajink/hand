"""Continuous-handoff eval for the anchor-sweep reorient, with its own control.

Two rollouts per design, and the second is the point:

  trained    Policy A -> the trained B, in the env B was trained in (anchor sweeping to
             `hold_ik` on schedule).
  zero-B     the same env with B's action forced to zero: the SCRIPTED carry riding Policy A's
             real delivery. Any claim that the policy reoriented the shaft is a claim about the
             gap between these two, not about the first number.

The anchor is ENV state, not policy state — a B trained with it and evaluated without it is out
of distribution the same way a residual-scale mismatch is (gotcha #13) — so both rollouts carry
the identical --hold-* block, and it must match what the run's config.yaml says.

    MUJOCO_GL=egl uv run --extra rl --extra gpu python scripts/eval_real_v1_anchor.py \
        --design rv03_narrowy_sp40 --policy-a <A.pt> --policy-b <B.pt>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/rl/videos/20260828_real_v1_anchor"


def run(design: str, a: Path, b: Path, tag: str, zero: bool, sweep_from: int,
        sweep_steps: int) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{design}_{tag}.mp4"
    cmd = [sys.executable, str(ROOT / "scripts/rl_demo_handoff_continuous.py"),
           "--policy-a", str(a), "--policy-b", str(b),
           "--morphology-run", str(ROOT / f"results/phase1/real_v1/{design}"),
           "--lift-delta", "0.10", "--open-finger-from-keyframe",
           "--hold-ctrl-from-keyframe", "hold_ik",
           "--hold-switch-from-sim-step", str(sweep_from),
           "--hold-switch-steps", str(sweep_steps),
           "--hold-switch-min-z", "0.08",
           "--output", str(out)]
    if zero:
        cmd.append("--zero-b")
    # Its OWN kernel cache: this runs while the next design is training, and a shared Warp
    # cache races and NaNs (CLAUDE.md, environment).
    env = dict(os.environ, WARP_CACHE_PATH=tempfile.mkdtemp(), MUJOCO_GL="egl")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400, env=env)
    health = {}
    hj = out.with_suffix(".health.json")
    if hj.exists():
        health = json.loads(hj.read_text())
    min_z = (health.get("metrics") or {}).get("min_z_hold")
    for ln in (r.stdout or "").splitlines():
        if "honest hold metric" in ln:
            for t in ln.split(":")[-1].split():
                try:
                    min_z = float(t)
                    break
                except ValueError:
                    continue
    return {"design": design, "variant": tag, "video": str(out) if out.exists() else None,
            "min_z": min_z, "health": health, "rc": r.returncode,
            "stderr_tail": (r.stderr or "")[-400:] if r.returncode else ""}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True)
    ap.add_argument("--policy-a", type=Path, required=True)
    ap.add_argument("--policy-b", type=Path, required=True)
    ap.add_argument("--sweep-from", type=int, default=600)
    ap.add_argument("--sweep-steps", type=int, default=550)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for tag, zero in (("trained", False), ("zeroB", True)):
        r = run(args.design, args.policy_a, args.policy_b, tag, zero,
                args.sweep_from, args.sweep_steps)
        rows.append(r)
        m = (r["health"] or {}).get("metrics") or {}
        print(f"{args.design:20} {tag:8} min_z {str(r['min_z']):>7} "
              f"peak_cos {str(m.get('peak_cos','—')):>7} held_cos {str(m.get('held_cos_tail','—')):>7} "
              f"touch {m.get('touch_frac','—')} force {m.get('tip_force','—')} "
              f"verdict {(r['health'] or {}).get('verdict','—')}")
        if r["rc"]:
            print("   FAILED:", r["stderr_tail"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
